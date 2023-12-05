import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from GCN import GCN
from cora import CoraData, tensor_from_numpy

# 超参数定义
LEARNING_RATE = 0.1  # 学习率
WEIGHT_DECAY = 5e-4  # 正则化系数
EPOCHS = 500  # 完整遍历训练集的次数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 设备 GPU/CPU

# 加载数据，并转换为torch.Tensor
dataset = CoraData().data
node_feature = dataset.x / dataset.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
tensor_x = tensor_from_numpy(node_feature, DEVICE)
tensor_y = tensor_from_numpy(dataset.y, DEVICE)
tensor_train_mask = tensor_from_numpy(dataset.train_mask, DEVICE)
tensor_val_mask = tensor_from_numpy(dataset.val_mask, DEVICE)
tensor_test_mask = tensor_from_numpy(dataset.test_mask, DEVICE)
normalize_adjacency = CoraData.normalization(dataset.adjacency)  # 规范化邻接矩阵

num_nodes, input_dim = node_feature.shape  # （N,D）
# 转换为稀疏表示 加速运算 节省空间
indices = torch.from_numpy(np.asarray([normalize_adjacency.row,
                                       normalize_adjacency.col]).astype('int64')).long()
values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
tensor_adjacency = torch.sparse.FloatTensor(indices, values,
                                            (num_nodes, num_nodes)).to(DEVICE)
# 模型定义：Model, Loss, Optimizer
model = GCN(input_dim).to(DEVICE)  # 如果gpu>1 用DataParallel()包裹 单机多卡 数据并行
criterion = nn.CrossEntropyLoss().to(DEVICE)  # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(),
                       lr=LEARNING_RATE,
                       weight_decay=WEIGHT_DECAY)  # Adam优化器


# 训练主体函数
def train():
    loss_history = []
    val_acc_history = []
    model.train()  # 训练模式
    train_y = tensor_y[tensor_train_mask]  # 训练节点的标签
    for epoch in range(EPOCHS):  # 完整遍历一遍训练集 一个epoch做一次更新
        logits = model(tensor_adjacency, tensor_x)  # 所有数据前向传播 （N,7）
        train_mask_logits = logits[tensor_train_mask]  # 只选择训练节点进行监督
        loss_gcn = criterion(train_mask_logits, train_y)  # 计算损失值
        optimizer.zero_grad()  # 清空梯度
        loss_gcn.backward()  # 反向传播计算参数的梯度
        optimizer.step()  # 使用优化方法进行梯度更新
        train_acc, _, _ = test(tensor_train_mask)  # 计算当前模型训练集上的准确率
        val_acc, _, _ = test(tensor_val_mask)  # 计算当前模型在验证集上的准确率
        # 记录训练过程中损失值和准确率的变化，用于画图
        loss_history.append(loss_gcn.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss_gcn.item(), train_acc.item(), val_acc.item()))
    plot_loss_with_acc(loss_history, val_acc_history)
    return loss_history, val_acc_history


# 测试函数
def test(mask):
    model.eval()  # 测试模式
    with torch.no_grad():  # 关闭求导
        logits = model(tensor_adjacency, tensor_x)  # 所有数据作前向传播
        test_mask_logits = logits[mask]  # 取出相应数据集对应的部分
        predict_y = test_mask_logits.max(1)[1]  # 按行取argmax 得到预测的标签
        accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()  # 计算准确率
    return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy()


# 可视化训练集损失和验证集准确率变化
def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')
    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


if __name__ == '__main__':
    loss, val = train()  # 每个 epoch 模型在训练集上的 loss 和 验证集上的准确率
    print("acc_val", sum(val)/len(val))
    # 计算最后训练好的模型在测试集上准确率
    test_acc, test_logits, test_label = test(tensor_test_mask)
    print("Test accuarcy: ", test_acc.item())
