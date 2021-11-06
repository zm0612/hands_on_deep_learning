import torch

# 自动求导的一个简单例子
x = torch.arange(4.0)
x.requires_grad_(True)  # 设置是否进行求梯度，requires_grad是返回变量是否求梯度
print(x.grad)

y = 2 * torch.dot(x, x)
print(y)

y.backward()
print(x.grad)

# 非标量变量的反向传播
x.grad.zero_()
y = x * x
y.sum().backward()
print(x.grad)

# 分离计算
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)
