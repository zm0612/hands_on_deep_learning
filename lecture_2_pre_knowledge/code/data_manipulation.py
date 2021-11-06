import torch

x = torch.arange(12)
print(x)

print(x.shape)  # tensor的形状
print(x.numel())  # tensor的元素个数

X = x.reshape(3, 4)
Y = x.reshape(-1, 4)
Z = x.reshape(3, -1)
print(X.size())  # 等价于shape

zero = torch.zeros((2, 3, 4))  # 全0矩阵
print(zero)

one = torch.ones((2, 3, 4))  # 全1矩阵

rand = torch.randn(3, 4)  # 标准高斯正太分布，均值为0，方差为1

tensor_list = torch.Tensor([[2, 1, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]])
print(tensor_list)

# 运算
x_0 = torch.tensor([1.0, 2, 4, 8])
y_0 = torch.tensor([2, 2, 2, 2])
print(x_0 + y_0)
print(x_0 - y_0)
print(x_0 * y_0)
print(x_0 / y_0)
print(x_0 ** y_0)

print(torch.exp(x_0))

x_1 = torch.arange(12, dtype=torch.float32).reshape((3, 4))
y_1 = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((x_1, y_1), dim=0))  # 顺着行往下拼接
print(torch.cat((x_1, y_1), dim=1))  # 顺着列往下拼接
print(x_1 == y_1)  # 对应位置进行比较

print(x_1.sum())

# 广播机制
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a + b)  # 将矩阵a复制列，将矩阵b复制行，完成广播机制

# 索引和切片
x_2 = torch.arange(12, dtype=torch.float32).reshape((3, 4))
print(x_2[-1])
print(x_2[1:3])

x_2[1, 2] = 9  # 赋值

x_2[0:2, :] = 12  # 第一行和第二行都赋值为12

# 节省内存
z_0 = torch.arange(12, dtype=torch.float32).reshape((3, 4))
z_0[:] = x_2 + x_2  # 切片索引可以避免内存重新申请
x_2 += x_2 + y_1  # +=也可以避免内存的重新申请

# 转换为其他python对象
A = x_2.numpy()
B = torch.tensor(A)
print(type(A))
print(type(B))

a = torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print(int(a))
