import numpy as np
import torch


# 根据最佳权重提取重要特征（脑区）
def feature_extraction(hgw1, hgw2, featurenum):
    hgw1_temp = hgw1.reshape(360, -1)  # 360*8960
    hgw1_fc = hgw1_temp.sum(1)
    hgw1_fc = abs(hgw1_fc)
    hgw1_temp2 = hgw1.reshape(360, -1, 180)
    w1mulw2 = hgw1_temp2.matmul(hgw2).reshape(360, -1)
    w1mulw2_fc = abs(w1mulw2).sum(1)
    features = w1mulw2_fc
    feat = abs(features) / 10
    a, b, c, d = np.array_split(feat, 4)
    feat = a + b + c + d

    B = np.argsort(feat)
    B = list(reversed(B))  # B中存储排序后的下标
    A = sorted(feat, reverse=True)  # A中存储排序后的结果
    AA = torch.tensor(A)
    BB = torch.tensor(B)
    return AA, BB
