from config.config import get_config
import scipy.io as scio
import numpy as np
#W是超边权重矩阵
def set_weight(H,fts):



    #fts = fts.reshape(cfg1['subjectnum'], -1)
    col = fts.shape[1]#col是特征数
    # 将关联矩阵倒置，此时行表示超边
    H_ = H.T
    # 找到倒置矩阵中值为1的位置
    A = np.where(H_!=0)#转换为数组,条件成立时，where返回的是每个符合condition条件元素的坐标.数据有多少维，结果就有多少维，每一维都是坐标。
    # 存放权重值
    W_ = []
    for i in range(fts.shape[0]):
        # 找到第i个超边的权重
        x = np.where(A[0]==i)#A[0]是超边下标维度，A[1]是顶点维度
        # 计算超边关联的顶点个数
        count = np.size(x)
        v = np.zeros((count, col))   #v是求超边中各顶点典型相关系数的辅助空间
        pccs = np.zeros(sum(range(1, count)))   #存储相关系数的临时变量
        # 根据上面的坐标找到列坐标对应位置的值，这个值就是超边对应的顶点号
        y = A[1][x]
        #print(y)
        # 找到每条超边所关联的count个顶点
        for j in range(count):
            v[j] = fts[y[j]]
        # 计算count个顶点两两皮尔逊相关系数
        for m in range(count):
            for n in range(m+1, count):
                if(np.isnan(np.min(np.corrcoef(v[m], v[n])))):
                    pccs[int(m * (count - 1) - (m * (m + 1)) / 2 + n - 1)]=0
                else:
                    pccs[int(m * (count - 1) - (m * (m + 1)) / 2 + n - 1)] = np.min(np.corrcoef(v[m], v[n]))  # 一个复杂的数组下标计算方式，其实用i++就行


        ###################################上述内容有效，防止因为数据缺失导致W计算错误
        # weightsum=0;
        # for k in range(len(H[1,:])):
        #     weightsum=weightsum+H[i,k]
        mean = np.mean(pccs)  #*weightsum/500
        W_.append(mean)
    W = np.diag(W_)


    return W