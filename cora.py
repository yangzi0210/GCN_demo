# 导入必要的库
import itertools
import os
import os.path as osp
import pickle
import urllib
from collections import namedtuple
import numpy as np
import scipy.sparse as sp  # 邻接矩阵用稀疏矩阵形式存储 节省空间
import torch

Data = namedtuple('Data', ['x', 'y', 'adjacency',
                           'train_mask', 'val_mask', 'test_mask'])


def tensor_from_numpy(x, device):  # 将数据从数组格式转换为tensor格式 并转移到相关设备上
    return torch.from_numpy(x).to(device)


class CoraData(object):
    # 数据集下载链接
    download_url = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"
    # 数据集中包含的文件名
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="cora", rebuild=False):
        """Cora数据，包括数据下载，处理，加载等功能
        当数据的缓存文件存在时，将使用缓存文件，否则将下载、进行处理，并缓存到磁盘
        处理之后的数据可以通过属性 .data 获得，它将返回一个数据对象，包括如下几部分：
            * x: 节点的特征，维度为 2708 * 1433，类型为 np.ndarray
            * y: 节点的标签，总共包括7个类别，类型为 np.ndarray
            * adjacency: 邻接矩阵，维度为 2708 * 2708，类型为 scipy.sparse.coo.coo_matrix
            * train_mask: 训练集掩码向量，维度为 2708，当节点属于训练集时，相应位置为True，否则False
            * val_mask: 验证集掩码向量，维度为 2708，当节点属于验证集时，相应位置为True，否则False
            * test_mask: 测试集掩码向量，维度为 2708，当节点属于测试集时，相应位置为True，否则False
        Args:
        -------
            data_root: string, optional
                存放数据的目录，原始数据路径: {data_root}/raw
                缓存数据路径: {data_root}/processed_cora.pkl
            rebuild: boolean, optional
                是否需要重新构建数据集，当设为True时，如果存在缓存数据也会重建数据
        """
        self.data_root = data_root
        save_file = osp.join(self.data_root, "processed_cora.pkl")
        if osp.exists(save_file) and not rebuild:  # 使用缓存数据
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self.maybe_download()  # 下载或使用原始数据集
            self._data = self.process_data()  # 数据预处理
            with open(save_file, "wb") as f:  # 把处理好的数据保存为缓存文件.pkl 下次直接使用
                pickle.dump(self.data, f)
            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        引用自：https://github.com/rusty1s/pytorch_geometric
        """
        print("Process data ...")
        # 读取下载的数据文件
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(
            osp.join(self.data_root, "raw", name)) for name in self.filenames]

        train_index = np.arange(y.shape[0])  # 训练集索引
        val_index = np.arange(y.shape[0], y.shape[0] + 500)  # 验证集索引
        sorted_test_index = sorted(test_index)  # 测试集索引

        x = np.concatenate((allx, tx), axis=0)  # 节点特征 N*D 2708*1433
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)  # 节点对应的标签 2708

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]  # 节点数/数据量 2708

        # 训练、验证、测试集掩码
        # 初始化为0
        train_mask = np.zeros(num_nodes, dtype=np.bool_)
        val_mask = np.zeros(num_nodes, dtype=np.bool_)
        test_mask = np.zeros(num_nodes, dtype=np.bool_)

        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True

        # 构建邻接矩阵
        adjacency = self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)  # （N*D）
        print("Node's label shape: ", y.shape)  # (N,)
        print("Adjacency's shape: ", adjacency.shape)  # (N,N)
        # 训练、验证、测试集各自的大小
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x, y=y, adjacency=adjacency,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    def maybe_download(self):
        # 原始数据保存路径
        save_path = os.path.join(self.data_root, "raw")
        # 下载相应的文件
        for name in self.filenames:
            if not osp.exists(osp.join(save_path, name)):
                self.download_data(
                    "{}/{}".format(self.download_url, name), save_path)

    @staticmethod
    def build_adjacency(adj_dict):
        """根据下载的邻接表创建邻接矩阵"""
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        # 去除重复的边
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        # 稀疏矩阵 存储非0值 节省空间
        adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                                   (edge_index[:, 0], edge_index[:, 1])),
                                  shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        """使用不同的方式读取原始数据以进一步处理"""
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out

    @staticmethod
    def download_data(url, save_path):
        """数据下载工具，当原始数据不存在时将会进行下载"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data = urllib.request.urlopen(url)
        filename = os.path.split(url)[-1]

        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(data.read())

        return True

    @staticmethod
    def normalization(adjacency):
        """计算 L=D^-0.5 * (A+I) * D^-0.5"""
        adjacency += sp.eye(adjacency.shape[0])  # 增加自连接 不仅考虑邻接节点特征 还考虑节点自身的特征
        degree = np.array(adjacency.sum(1))  # 此时的度矩阵的对角线的值 为 邻接矩阵 按行求和
        d_hat = sp.diags(np.power(degree, -0.5).flatten())  # 对度矩阵对角线的值取-0.5次方 再转换为对角矩阵
        return d_hat.dot(adjacency).dot(d_hat).tocoo()  # 归一化的拉普拉斯矩阵 稀疏存储 节省空间
