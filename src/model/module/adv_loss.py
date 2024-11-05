"""
这里用 PyTorch 的 Function 类定义了与 TensorFlow 代码功能相同的 FlipGradient 类。
在 forward() 方法中，我们接收输入张量 x 和权重值 l，将 x 作为正向传播的输出，
并将 l 存储在上下文中以便于反向传播。
在 backward() 方法中，我们计算梯度反向传播时的负梯度乘上权重 l，返回给反向传播。

为了与 TensorFlow 代码中的 RegisterGradient 函数相匹配，
我们还定义了一个 flip_gradient() 函数，
它仅接收输入张量 x 和权重值 l。
在这个函数中，我们使用 PyTorch 中的 FlipGradient 类，
并将输入 x 和权重 l 传递给其 apply() 方法来获得正向传播的结果。
https://github.com/thecharm/AGBAN/blob/main/main.py
"""

import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
# import random

class FlipGradient(Function):
    @staticmethod
    def forward(ctx, x, l=1.0):
        ctx.l = l
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output.neg() * ctx.l), None

def flip_gradient(x, l=1.0):
    return FlipGradient.apply(x, l)


class Adversarial_loss(nn.Module):
    def __init__(self, input_dim, advloss_dropout):
        super(Adversarial_loss, self).__init__()
        self.input_dim = input_dim
        self.task_num = 2
        self.adv_dropout = nn.Dropout(advloss_dropout)
        self.linear = nn.Linear(input_dim, self.task_num)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_shared_feature, task_labels):
        num_steps = input_shared_feature.size(1)
        shared_output1 = input_shared_feature.unsqueeze(1) #adding the channel
        max_pool_output = F.max_pool2d(shared_output1, kernel_size=[num_steps, 1], stride=[num_steps, 1],  padding=(0, 0))
        max_pool_output = max_pool_output.view(-1, self.input_dim)

        feature = flip_gradient(max_pool_output)
        feature = self.adv_dropout(feature)
        logits = F.relu(self.linear(feature))
        # logits = self.linear(feature)
        adv_loss = F.cross_entropy(logits, task_labels)
        return adv_loss

    def multi_task(self, shared_output):
        num_steps = shared_output.size(1)
        shared_output1 = shared_output.unsqueeze(1) #adding the channel
        max_pool_output = F.max_pool2d(shared_output1, kernel_size=[num_steps, 1], stride=[num_steps, 1],  padding=(0, 0))
        max_pool_output = max_pool_output.view(-1, self.input_dim)
        return max_pool_output

if __name__ == "__main__":
    setting = type('', (), {})()
    setting.lstm_dim = 8
    setting.num_steps = 2
    setting.task_num = 2
    drop_out = 0.7
    input_dim = 8
    # setting.keep_prob1 = 0.7
    # setting.adv_weight = 0.06
    # torch.manual_seed(44)
    input_tensor = torch.randn(1, setting.num_steps, setting.lstm_dim)  # (batch_size, num_steps, lstm_dim)
    # input_tensor = torch.tensor([[[ 1.7516, 1.7885,  0.9315, -0.7431, -0.7264, -1.1203,  1.1608, -0.8058],
    #                               [ 0.9482, -1.5415, 0.3986,  0.474,  1.17,   -0.4089, -2.2787, -1.3553]]])
    task_label = torch.randn([1, 2]).int()  # (batch_size, task_num)
    # task_label = torch.tensor([[0, 1]])
    # print(task_label)
    model = Adversarial_loss(input_dim, drop_out, 1)
    feature = model.multi_task(input_tensor)
    # print(feature)
    loss = model(feature, task_label)
    print(loss)