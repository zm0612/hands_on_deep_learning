import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])
print(x + y)  # 对应的+、-、×、/、**

x = torch.arange(4)
print(x[3])

# 长度、维度和形状
print(len(x))
print(x.shape)

# 矩阵
A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)  # A转置

# 张量
X = torch.arange(24).reshape(2, 3, 4)
print(X)

# 张量算法的基本性质
B = torch.arange(20).reshape(5, 4)
print(A * B)  # 哈达玛积，对应元素相乘，不是矩阵的惩罚

# 矩阵降维
C = torch.arange(20).reshape(5, 4)
print(C.sum(axis=0))  # 按行进行求和，得到4维的向量
print(C.sum(axis=1))  # 按列就行求和，得到5维的向量

# 非降维求和
print(C.sum(axis=0, keepdims=True))  # 保持维度不变，二维矩阵求和之后还是二维矩阵，但是只有1x5

# 点积
D = torch.arange(5)
E = torch.arange(5)
print(torch.dot(D, E))  # 对应元素相乘，然后求和

# 矩阵和向量相乘
F = torch.arange(25).reshape(5, 5)
G = torch.arange(5)
print(torch.mv(F, G))

# 矩阵和矩阵相乘
H = torch.arange(25).reshape(5, 5)
I = torch.arange(25).reshape(5, 5)
print(torch.mm(H, I))

# 范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))  # 二范数
print(torch.abs(u).sum())  # 一范数
