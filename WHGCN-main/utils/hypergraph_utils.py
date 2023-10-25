import numpy
import numpy as np


def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


# 相关距离
def CD(x):
    dist_mat = np.zeros((x.shape[0], x.shape[0]))
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[0]):
            d = 1 - abs(np.corrcoef(x[i], x[j]))
            dist_mat[i][j] = d[0][1]
    return dist_mat
    # 余弦距离


def Cosine(x):
    dist_mat = np.zeros((x.shape[0], x.shape[0]))
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[0]):
            d = np.dot(x[i], x[j]) / (np.linalg.norm(x[i]) * (np.linalg.norm(x[j])))
            dist_mat[i][j] = d
    return dist_mat


# 传过来的是一个特征列表
def feature_concat(*F_list, normal_col=False):
    """
    串联多模态特性。如果特征矩阵的维数大于2， 函数将其化简为二维(将最后一个维度作为特征维度， 另一个维度将被融合为对象维度)
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature    每一列是否规范化规范化
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])  # 我感觉这句有问题，按照列对齐
            # normal each column
            if normal_col:  # 如果没有正规化，则进行正规化
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix

            if features is None:
                features = f
                # 其实这里是没用的，因为这个代码并没有使用这里进行特征融合
            else:
                features = np.hstack((features, f))  # 将参数元组的元素数组按水平方向进行叠加
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max  # 好像没用
    return features


def hyperedge_concat(*H_list):  # 传输过来的应该是一个H矩阵的列表
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, W, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, W, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, W, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """

    H = np.array(H)
    DV = np.sum(H * W, axis=1)  # 得到一列顶点权重
    DE = np.sum(H, axis=0)  # 超边的度的行向量

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))  # np.power是求x的y次幂
    # W = np.mat(np.diag(W))
    H = np.mat(H)  # np.mat用于将输入解释为矩阵
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
        return DV2_H, invDE_HT_DV2
    else:
        # for i in range(invDE.shape[0]):
        #     for j in range(invDE.shape[1]):
        #         if numpy.isnan(invDE[i, j]):
        #             invDE[i, j] = 0.0001
        # for i in range(DV2.shape[0]):
        #     for j in range(DV2.shape[1]):
        #         if numpy.isnan(DV2[i, j]):
        #             DV2[i, j] = 0.0001

        G = DV2 * H * invDE * W.T * HT * DV2
        I = np.identity(G.shape[0], float)
        G = I + G
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):  # 根据距离矩阵构建超图H
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()  # 将dis_vec中的元素从小到大排列，提取其对应的index(索引)，
        # squeeze()，假如某一维只有一项数据，则删除这一维度。
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)
                # 这里有问题
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs, split_diff_scale, is_probH, m_prob, subjectnum):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    是否在不同的邻居规模上超边

    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """

    # if len(X.shape) != 2:
    #     X = X.reshape(-1, X.shape[-1])            #按照列对齐
    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    # 欧式距离
    dis_mat = Eu_dis(X)
    # 相关距离
    # dis_mat =CD(X)
    # 余弦距离
    # dis_mat = Cosine(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)  # append命令是将整个对象加在列表末尾，就是直接拼接上

    # shape[0]为行数，shape[1]为列数
    for i in range(X.shape[0]):
        if numpy.isnan(X[i][0]):
            for j in range(X.shape[0]):
                H[i][j] = 0
    # print(H)
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[0]):
    #         if (numpy.isnan(H[i][j])):
    #             print("H为nan")
    return H


def statistic_indicators(output, labels):
    TP = ((output == 1) & (labels == 1)).sum()
    TN = ((output == 0) & (labels == 0)).sum()
    FN = ((output == 0) & (labels == 1)).sum()
    FP = ((output == 1) & (labels == 0)).sum()
    return TP, TN, FN, FP
