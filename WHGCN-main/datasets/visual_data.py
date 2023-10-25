from datasets import load_ft
from utils import hypergraph_utils as hgut
import numpy as np


# 根据多种模态的特征分别构建超图
def load_feature_construct_H(data_dir,
                             subjectnum,
                             featurenum,
                             positiveclass,
                             negativeclass,
                             m_prob=1,
                             K_neigs=None,
                             is_probH=True,
                             split_diff_scale=False
                             ):
    """

    :param data_dir: directory of feature data
    :param m_prob: parameter in hypergraph incidence matrix construction
    :param K_neigs: the number of neighbor expansion
    :param is_probH: probability Vertex-Edge matrix or binary
    :return:
    """
    # init feature
    ft, lbls, idx_train, idx_test = load_ft(data_dir, positiveclass, negativeclass)  # 从.mat文件读入数据，并且得到标签，还生成了训练集和测试集。
    # construct feature matrix
    fts = None
    fts = hgut.feature_concat(fts, ft)#将两个矩阵串联，其实就是放进去。
    if fts is None:
        raise Exception(f'None feature used for model!')#该模态没有数据

    # construct hypergraph incidence matrix
    print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
    H = None

    H = hgut.construct_H_with_KNN(ft, K_neigs=K_neigs,
                                    split_diff_scale=split_diff_scale,
                                    is_probH=is_probH, m_prob=m_prob,
                                    subjectnum=subjectnum)
    # H = hgut.hyperedge_concat(H, tmp)
    H[np.where(H != 0)] = 1
    print("数据维度fts:", ft.shape)
    print("标签数lbls:", lbls.shape)
    print("H的大小为:", H.shape)
    return ft, lbls, idx_train, idx_test, H
