from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import math

class Highway(nn.Module):
    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x


class GateConcMechanism(nn.Module):
    def __init__(self, hidden_size=None):
        super(GateConcMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.w1 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.w2 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.w1.size(1))
        stdv2 = 1. / math.sqrt(self.w2.size(1))
        stdv = (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input, hidden):
        # input: hidden state from encoder; hidden: hidden state from key value memory network
        # output = [gate * input; (1 - gate) * hidden]
        gated = input.matmul(self.w1.t()) + hidden.matmul(self.w2.t()) + self.bias
        gate = torch.sigmoid(gated)
        output = torch.cat([input.mul(gate), hidden.mul(1 - gate)], dim=2)
        return output


class LinearGateAddMechanism(nn.Module):
    def __init__(self, hidden_size=None):
        super(LinearGateAddMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.w1 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.w2 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.w1.size(1))
        stdv2 = 1. / math.sqrt(self.w2.size(1))
        stdv = (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden):
        # input: hidden state from encoder; hidden: hidden state from key value memory network
        # output = [gate * input; (1 - gate) * hidden]
        input = self.linear1(input)
        hidden = self.linear2(hidden)
        gated = input.matmul(self.w1.t()) + hidden.matmul(self.w2.t()) + self.bias
        gate = torch.sigmoid(gated)
        output = input.mul(gate) + hidden.mul(1 - gate)
        return output

style_map = {
    'add': lambda x, y: x + y,
    'concat': lambda *args: torch.cat(args, args[0].dim() - 1),
    'diff': lambda x, y: x - y,
    'abs-diff': lambda x, y: torch.abs(x - y),
    'concat-diff': lambda x, y: torch.cat((x, y, x - y), x.dim() - 1),
    'concat-add': lambda x, y: torch.cat((x, y, x + y), x.dim() - 1),
    'concat-abs-diff': lambda x, y: torch.cat((x, y, torch.abs(x - y)), x.dim() - 1),
    'mul': lambda x, y: torch.mul(x, y),
    'concat-mul-diff': lambda x, y: torch.cat((x, y, torch.mul(x, y), torch.abs(x - y)), x.dim() - 1)
}

class FusionModule(nn.Module):
    """
    FusionModule定义了encoder output与kv output之间的信息融合方式
    """
    def __init__(self, layer=1, fusion_type="concat", input_size=1024, output_size=1024, dropout=0.2):
        """
        :param layer: layer代表highway的层数
        :param fusion_type: fusion_type代表融合方式
        :param size: size代表输出dimension
        :param dropout: 代表fusion之后，highway之前的dropout
        """
        super(FusionModule, self).__init__()
        self.fusion_type = fusion_type
        self.layer = layer
        if self.layer > 0:
            self.highway = Highway(size=output_size, num_layers=layer, f=torch.nn.functional.relu)
        # if self.fusion_type == "gate-add":
        #     self.gate = GateAddMechanism(hidden_size=input_size)
        elif self.fusion_type == "gate-concat":
            self.gate = GateConcMechanism(hidden_size=input_size)
        elif self.fusion_type == "l-gate-add":
            self.gate = LinearGateAddMechanism(hidden_size=input_size)
        self.fusion_dropout = nn.Dropout(p=dropout)

    def forward(self, enc_out, kv_out):
        # 如果使用gate的方式进行fusion
        if self.fusion_type in ["gate-add", "gate-concat", "l-gate-add"]:
            fused = self.gate(enc_out, kv_out)
        # 直接用concat或者add等方式进行fusion
        else:
            fused = style_map[self.fusion_type](enc_out, kv_out)
        fused = self.fusion_dropout(fused)
        # 进行highway操作
        if self.layer > 0:
            fused = self.highway(fused)
        return fused