import scipy.io as scio
import numpy as np
# 根据目录加载特征数据，并划分训练集和测试集


def load_ft(data_dir, positiveclass, negativeclass):
    data = scio.loadmat(data_dir)
    lbls = np.concatenate((np.zeros(positiveclass), np.ones(negativeclass))).astype(np.longlong)
    # label是1维向量，positiveclass个0，和negativeclass个1.
    shuffle_idx = np.array(range(0, lbls.shape[0]))  # 创建一个数组，计数从 0开始。计数到 stop 结束，不包含stop。
    # 作用就是重新排序返回一个随机序列作用类似洗牌
    np.random.shuffle(shuffle_idx)
    feat = data['feat']
    # .mat文件中的变量必须叫feat
    feat = feat.reshape(lbls.shape[0], -1)
    fts = feat.astype(np.float32)
    n = int(lbls.shape[0]*0.8)   # 划分训练集和测试集
    idx_train = shuffle_idx[0:n]
    idx_test = shuffle_idx[n:lbls.shape[0]]
    # idx_train = np.where(idx == 0)[0]
    # idx_test = np.where(idx == 1)[0]
    print("idx_train:", idx_train.shape)
    print("idx_test:", idx_test.shape)
    return fts, lbls, idx_train, idx_test
# 返回全部数据，标签，还有训练集和测试集的序号。
