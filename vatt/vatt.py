from typing import (
    List
)
import torch as T
import math
from torch import nn
from torch.nn import functional as F
from transformers.models.bert.modeling_bert import BertSelfOutput, BertOutput, BertAttention, BertLayer
from AdaptersComponents.AdapterModules import AdapterModule

def split_dim(x: T.Tensor, n: int, *, dim: int=-1):
    assert -x.ndim <= dim < x.ndim
    assert x.shape[dim] % n == 0
    if dim < 0:
        dim = dim + x.ndim
    sh = x.shape[:dim] + (x.shape[dim] // n, n) + x.shape[dim+1:]
    z = x.view(sh)
    return z

class EnhancedLinear(nn.Module):
    '''"Stop thinking with your head. -- SMerity'''
    def __init__(
            self,
            Din: int,
            Dout: int,
    ):
        self.Din = Din
        self.Dout = Dout
    
        self.static = nn.Parameter(T.ones(self.Dout), requires_grad=True)
        self.linear = nn.Linear(self.Din, self.Dout * 2)

    def forward(self, X: T.Tensor) -> T.Tensor:
        mag, sgn = self.linear(X).chunk(2, dim=-1)
        Z = self.static * mag.sigmoid() * sgn.tanh()
        return Z

class SelfAttention(nn.Module):
    def __init__(self, args: dict, config):
        super().__init__()

        self.args = args
        self.config = config
        self.use_common_transform = args['vatt-use-common-transform']
        self.nb_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.T = 1
        # self.reduction = self.T / 1000.0
        self.reduction = 0.0
        self.use_adapter = args['vatt-final-adapter']
        self.positional_keys_mode = args['vatt-positional-keys']
        self.do_debug_shapes = args.get('vatt-debug-shapes', False)

        self.build_()

    def build_key_transform_(self):
        args = self.args
        config = self.config
        if self.positional_keys_mode:
            key_size = self.nb_layers * 2
            if self.positional_keys_mode == 'random':
                self.StaticKeys = [
                    T.rand(key_size).cuda()
                    for _ in range(self.nb_layers)
                ]
            elif self.positional_keys_mode == 'sinosoid':
                self.StaticKeys = []
                di = T.arange(key_size)
                di = (di // 2).type(T.float32) / key_size
                di = T.div(1.0, T.pow(100, di))
                for ilayer in range(self.nb_layers):
                    enc = di * ilayer
                    enc[0::2].sin_()
                    enc[1::2].cos_()
                    self.StaticKeys.append(enc.cuda())
            else:
                raise KeyError(f"Unknown positional key mode: {self.positional_keys_mode}.")
            shared_key_transform = EnhancedLinear(key_size, self.hidden_size)
            self.key_transforms = nn.ModuleList([
                shared_key_transform
                for _ in range(self.nb_layers)
            ])
        else:
            self.key_transforms = nn.ModuleList([
                nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.nb_layers)
            ])

    def build_(self):
        args = self.args
        config = self.config
        self.dropout = nn.Dropout(0.1)
        self.query   = nn.Linear(self.hidden_size, self.hidden_size)
        if self.use_common_transform:
            self.key_c   = nn.Linear(self.hidden_size, self.hidden_size)
            self.value_c = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.build_key_transform_()
        self.value_transforms   = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            for _ in range(self.nb_layers)
        ])
        self.adapter = AdapterModule(self.hidden_size, args['vatt-bottleneck_dim'])
        self.values_lnorm = nn.LayerNorm(self.hidden_size)

    def debug_shapes(self, tensor, name: str):
        if self.do_debug_shapes:
            print(f'{name}.shape=={tensor.shape}')

    def Tquery(self, query: T.Tensor) -> T.Tensor:
        #^ query: [batch, dim]
        #^ => [b, Qdim]
        return self.query(query)

    def Tkeys(self, Xs: List[T.Tensor]) -> T.Tensor:
        if self.positional_keys_mode:
            Xs = self.StaticKeys
            #^ Xs: [layer][KeySize]
            Xs = [xs.unsqueeze(0) for xs in Xs]
            #^ Xs: [layer][1, KeySize]
        Z = [
            self.key_c(tkey(xs))
            if self.use_common_transform
            else tkey(xs)
            for tkey, xs in zip(self.key_transforms, Xs)
        ]
        return T.stack(Z, dim=1)
        #^ => [b|1, layer, Qdim]

    def Tvalues(self, Xs: List[T.Tensor]) -> T.Tensor:
        #^ => [b, Vdim, layer]
        if self.use_common_transform:
            Z = [
                self.value_c(tlvalue(xs))
                for tlvalue, xs in zip(self.value_transforms, Xs)
            ]
        else:
            Z = [
                tlvalue(xs)
                for tlvalue, xs in zip(self.value_transforms, Xs)
            ]
        return T.stack(Z, dim=2)

    def forward(self, Xs: List[T.Tensor], Q: T.Tensor) -> T.Tensor:
        #^ => [b, Vd] or [b, AdapterD]
        #^ Xs: [layer][batch, dim]
        #^ Q: [batch, dim]
        Xs = [self.dropout(x) for x in Xs]

        query = self.Tquery(Q)
        #^ query: [batch, Qdim]
        keys = self.Tkeys(Xs)
        #^ keys: [batch, layer, Qdim]
        values = self.Tvalues(Xs)
        #^ values: [batch, Vdim, layer]
        # residual = Xs[-1]
        #^ residual: [batch, Vdim]

        # values += residual.unsqueeze(2)
        #^ values: [batch, Vdim, layer]

        # values = self.dropout(values)
        values = self.values_lnorm(values.transpose(-1, -2)).transpose(-1, -2)
        #^ => [batch, layer, Vdim] => [batch, norm(Vdim), layer]

        # query_layer = self.query(query)
        # key_layer = self.key_transforms(key)
        # value_layer = self.value_transforms(value)

        attention_logits = T.matmul(query.unsqueeze(1), keys.transpose(-1, -2)).squeeze(dim=1)
        #^ [b, 1, Qd] matmul [b, (Qd, L)] => [b, L]

        attention_logits = self.dropout(attention_logits)

        attention_probs = F.softmax(attention_logits / math.sqrt(self.hidden_size), dim=-1)
        # attention_probs = nn.Softmax(dim=-1)(attention_logits / math.sqrt(self.hidden_size) / self.T)
        # self.T = max(self.T - self.reduction, 1.0)

        context_layer = T.matmul(attention_probs.unsqueeze(1), values.transpose(-1, -2)).squeeze(dim=1)
        #^ [b, 1, L] matmul [b, (L, Vd)] => [b, Vd]

        # context_layer += self.value_transforms[-1](residual)

        if self.use_adapter: 
            context_layer = self.adapter(context_layer)
        return context_layer