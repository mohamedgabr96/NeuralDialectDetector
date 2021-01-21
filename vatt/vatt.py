from typing import (
    List
)
import torch as T
import math
from torch import nn
from transformers.models.bert.modeling_bert import BertSelfOutput, BertOutput, BertAttention, BertLayer
from AdaptersComponents.AdapterModules import AdapterModule


class SelfAttention(nn.Module):
    def __init__(self, args: dict, config):
        super().__init__()

        self.args = args
        self.config = config
        self.use_common_transform = args['vatt-use-common-transform']
        self.nb_layers = config.num_hidden_layers
        self.T = 1
        self.reduction = self.T / 1000.0
        self.use_adapter = args['vatt-final-adapter']
        self.is_keys_positional = args['vatt-positional-keys']

        self.build()

        self.do_debug_shapes = args.get('vatt-debug-shapes', False)

    def build(self):
        args = self.args
        config = self.config
        self.dropout = nn.Dropout(0.1)
        self.dense   = nn.Linear(config.hidden_size, 1)
        self.query   = nn.Linear(config.hidden_size, config.hidden_size)
        if self.use_common_transform:
            self.key_c   = nn.Linear(config.hidden_size, config.hidden_size)
            self.value_c = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        if self.is_keys_positional:
            key_size = self.nb_layers + 6
            self.StaticKeys = [
                T.rand(key_size).cuda()
                for _ in range(self.nb_layers)
            ]
            shared_key_transform = nn.Linear(key_size, config.hidden_size)
            self.key_transforms = nn.ModuleList([
                shared_key_transform
                for _ in range(self.nb_layers)
            ])
        else:
            self.key_transforms = nn.ModuleList([
                nn.Linear(config.hidden_size, config.hidden_size)
                for _ in range(self.nb_layers)
            ])
        self.value_transforms   = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(self.nb_layers)
        ])
        self.adapter = AdapterModule(config.hidden_size, args['vatt-bottleneck_dim'])


    def debug_shapes(self, tensor, name: str):
        if self.do_debug_shapes:
            print(f'{name}.shape=={tensor.shape}')

    def Tquery(self, query: T.Tensor) -> T.Tensor:
        #^ query: [batch, dim]
        #^ => [b, Qdim]
        return self.query(query)

    def Tkeys(self, Xs: List[T.Tensor]) -> T.Tensor:
        if self.is_keys_positional:
            Xs = self.StaticKeys
            #^ Xs: [layer][KeySize]
            Xs = [xs.unsqueeze(0) for xs in Xs]
            #^ Xs: [layer][1, KeySize]
        if self.use_common_transform:
            Z = [
                self.key_c(tkey(xs))
                for tkey, xs in zip(self.key_transforms, Xs)
            ]
        else:
            Z = [
                tkey(xs)
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
        query = self.Tquery(Q)
        self.debug_shapes(query, name='query')
        #^ query: [batch, Qdim]
        keys = self.Tkeys(Xs)
        self.debug_shapes(keys, name='keys')
        #^ keys: [batch, layer, Qdim]
        values = self.Tvalues(Xs)
        self.debug_shapes(values, name='values')
        #^ values: [batch, Vdim, layer]
        residual = Xs[-1]
        self.debug_shapes(residual, name='residual')
        #^ residual: [batch, Vdim]
    
        values += residual.unsqueeze(2)

        # query_layer = self.query(query)
        # key_layer = self.key_transforms(key)
        # value_layer = self.value_transforms(value)

        attention_scores = T.matmul(query.unsqueeze(1), keys.transpose(-1, -2)).squeeze(dim=1)
        self.debug_shapes(attention_scores, name='attention_scores')
        #^ [b, 1, Qd] matmul [b, (Qd, L)] => [b, L]

        attention_scores = self.dropout(attention_scores)

        attention_probs = nn.Softmax(dim=-1)(attention_scores / math.sqrt(self.nb_layers) / self.T)
        self.T = max(self.T - self.reduction, 1.0)

        context_layer = T.matmul(attention_probs.unsqueeze(1), values.transpose(-2, -1)).squeeze(dim=1)
        self.debug_shapes(context_layer, name='context_layer')
        #^ [b, 1, L] matmul [b, (L, Vd)] => [b, Vd]

        if self.use_adapter: 
            context_layer = self.adapter(context_layer)
        return context_layer