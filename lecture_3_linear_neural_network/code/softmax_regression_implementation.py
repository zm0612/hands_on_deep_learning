import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# X, y = next(iter(train_iter))
# print(X.shape)

num_inputs = 784  # 图像是28x28
num_outputs = 10  # 一共10个类别

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X.sum(0, keepdim=True))
print(X.sum(1, keepdim=True))


# softmax实现
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 广播机制，对partition的列进行复制


X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob)
print(X_prob.sum(1))


def net(X):
    # 将X的形状进行reshape，目的是为了可以和W进行矩阵运算
    o = torch.matmul(X.reshape(-1, W.shape[0]), W) + b
    return softmax(o)


y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y])


# 交叉熵的函数定义
def cross_entropy(y_hat, y):
    # range(len(y_hat))生成一个范围列表，y对应的是真实标签
    # 由于只有对应真实标签时，y=1，其实均为0，所以交叉熵的计算只需要计算真实标签
    return -torch.log(y_hat[range(len(y_hat)), y])


print(cross_entropy(y_hat, y))


# 分类准确率
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


print(accuracy(y_hat, y) / len(y))


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


print(evaluate_accuracy(net, test_iter))


# 训练
def train_epoch_ch3(net, train_iter, loss, updater):
    """
    对softmax回归进行一次训练
    :param net: 网络结构
    :param train_iter: 训练数据迭代器
    :param loss: 损失函数计算方法
    :param updater: 网络参数更新
    :return:
    """
    if isinstance(net, torch.nn.Module):  # 是否当前网络是torch.nn.Module类型
        net.train()  # 将网络设置为训练模式
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        # 将数据输入网络进行一次传播
        y_hat = net(X)  # 在torch中使用dataloader加载图像时已经将其转换成tensor: 256x1x28x28
        l = loss(y_hat, y)  # 计算loss
        if isinstance(updater, torch.optim.Optimizer):  # 是否为torch中的优化器
            updater.zero_grad()  # torch中变量会保留上一次的梯度，所以每次都要清零
            l.backward()  # 执行反向传播计算每一个参数的梯度
            updater.step()  # 执行参数的一步更新，在构造updater时已经设置了更新率
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size().numel())
        else:
            l.sum().backward()  # 对loss进行求和，然后反向传播求梯度，torch中的计算图会自动计算需要求梯度的参数
            updater(X.shape[0])  # 对参数使用进行一步更新
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """
    训练网络
    :param net: 网络结构
    :param train_iter: 训练数据迭代器
    :param test_iter: 测试数据迭代器
    :param loss: 损失函数
    :param num_epochs: 训练次数
    :param updater: 网络参数更新器
    :return:
    """
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert 1 >= train_acc > 0.7, train_acc
    assert 1 >= test_acc > 0.7, test_acc


lr = 0.1


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


num_epochs = 10
# 对玩网络进行训练
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


def predict_ch3(net, test_iter, n=6):
    """预测标签（定义见第3章）。"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


# 进行网络测试
predict_ch3(net, test_iter)

d2l.plt.show()
