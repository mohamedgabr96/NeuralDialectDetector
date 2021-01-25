from typing import (
    Union,
    Dict,
)
import math
import torch as T
from torch import nn
from torch.nn import functional as F

class PlainDropout(nn.Dropout):
    def forward(self, X: T.Tensor, y: T.Tensor):
        return super()(X)

class MappedDropout(nn.Module):
    def __init__(
            self,
            drop_rates: T.Tensor,
    ):
        '''
        ^ drop_rates: [nClasses]=>p[c]
        '''
        super().__init__()
        self.keep_rates: T.Tensor = 1. - drop_rates

    def forward(self, X: T.Tensor, y: T.Tensor):
        rates = self.keep_rates[y].unsqueeze(1)
        #^ rates: [b, 1] float
        Xshape = X.shape
        n_batch = Xshape[0]
        X = X.view(n_batch, -1)
        mask = rates.expand_as(X).bernoulli()
        X = X * mask / rates
        return X.view(Xshape)

class BalancedDropout(nn.Module):
    def __init__(
            self,
            drop_rates: Union[float, T.Tensor]=0.5,
    ):
        '''
        ^ drop_rates: [nClasses]=>p[c]
        '''
        super().__init__()
        self.drop_rates = drop_rates
        if isinstance(drop_rates, float):
            self.drop = PlainDropout(drop_rates)
        elif isinstance(drop_rates, dict):
            self.drop = MappedDropout(drop_rates)

    def forward(self, X, y):
        return self.drop(X, y)

def minmax_norm(x: T.Tensor) -> T.Tensor:
    x = (x.max() - x) / (x.max() - x.min())
    x = 1.0 - x
    return x

def softmax_sqrt_temp_norm(x: T.Tensor, temp='auto') -> T.Tensor:
    if temp == 'auto':
        tt = math.sqrt(len(x))
    elif isinstance(temp, float):
        tt = math.sqrt(temp)
    x = x / tt
    x = F.softmax(x)
    return x

def direct_freq_dropout(freqs: T.Tensor, min_p=0.1, max_p=0.5, norm=minmax_norm):
    #^ freqs: [nClasses] int (freq of each class)
    x = freqs
    x = norm(x)
    rates = min_p + (max_p - min_p) * x
