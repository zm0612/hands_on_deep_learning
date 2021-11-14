import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# nn.Flatten()对输入的二维数据展平成一个向量
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weight)

# 选择交叉熵作为损失函数
loss = nn.CrossEntropyLoss()

# 构造梯度求解器
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练
num_epochs = 50
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

d2l.plt.show()
