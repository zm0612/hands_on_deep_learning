import random
import torch
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
batch_size = 10
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        # yield就是return返回一个值，并且记住这个返回的位置，下次迭代next就从这个位置后开始
        # 这样做的好处是不用把所有的内容都加载到内存中，用一些加载一些
        yield features[batch_indices], labels[batch_indices]


# 输出一个batch的数据
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

# 初始化模型参数
# w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
w = torch.zeros(size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 训练
lr = 0.003
num_epochs = 3
net = linreg
loss = squared_loss

# 整个训练的思路为：取出一个batch_size的数据对参数进行更新，然后直到数据全部使用一遍，就算完成了一次epoch
for epoch in range(num_epochs):
    # 取出10个数据, 对参数进行更新, 相当于更新100次
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():  # 后面的内容不进行计算图构建
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计值: {w}, w的真实值：{true_w}')
print(f'b的估计值: {b}, b的真实值：{true_b}')
