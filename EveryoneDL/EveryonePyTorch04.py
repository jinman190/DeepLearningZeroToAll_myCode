# 모두의 파이토치 첫 예제(처음 파이토치 씀, 역전파 해서 미분값 구하기만 파이토치 이용)
import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]), requires_grad=True)

def forward(x):
    return x*w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        w.data = w.data - 0.01 * w.grad.data
        print("grad: ", w.grad.data[0])
        w.grad.data.zero_()

    print("progress: ", epoch, l.data[0])

