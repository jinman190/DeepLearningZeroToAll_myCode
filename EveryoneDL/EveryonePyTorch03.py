# 모두의 파이토치 첫 예제(파이토치 안 씀, 미분 계산 해서 최저 loss 찾기)
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return x*w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

def gradient(x, y):
    return 2*x*(x*w-y)

for epoch in range(100):
    print("w=", w)
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        print("\tgrad:", x_val, y_val, grad)
        l = loss(x_val, y_val)

    print("progress:", epoch, " w=", w, " loss=", l)
