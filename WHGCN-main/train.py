import copy
import os
import feature_ex
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import utils.hypergraph_utils
import weight
from sklearn.metrics import roc_auc_score
from config.config import get_config
from datasets import load_feature_construct_H
from models import HGNN

mylog = open('result.log', mode='a', encoding='utf-8')

# num_epochs是最大迭代次数
# scheduler主要是为了在训练中以不同的测略来调整学习率，当然如果不涉及到学习率大小的变化也就用不到该函数了。
# print_freq是结果隔几轮输出一次结果
def train_model(model, criterion, optimizer, scheduler, num_epochs, print_freq):
    # 初始化模型参数和准确率
    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_acc = 0.0
    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs}')
        # Each epoch has a training and validation phase
        # optimizer.step()通常用在每个mini-batch之中，而scheduler.step()通常用在epoch里面,
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                idx = idx_train
                model.train()  # Set model to training mode
            else:
                idx = idx_test
                model.eval()  # Set model to evaluate mode评估模式
            # 初始化当前这一轮的损失和
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.遍历数据
            # 意思是把梯度置零
            optimizer.zero_grad()
            # with 提供了一种机制，可以在进入和退出（无论正常退出，还是异常退出）某个语句块时，自动执行自定义的代码。
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(feat, G)
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(outputs, 1)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            with torch.set_grad_enabled(phase == 'val'):
                outputs = model(feat, G)
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(outputs, 1)

            # statistics统计这一轮的结果
            running_loss += loss.item()
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])
            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'val':
                loss_list.append(float(epoch_loss))
                acc_list.append(float(epoch_acc))
            # 保持测试损失和准确率
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                TP, TN, FN, FP = utils.hypergraph_utils.stastic_indicators(outputs[idx_test], lbls[idx_test])
                ACC = (TP + TN) / (TP + TN + FP + FN)
                SEN = TP / (TP + FN)
                SPE = TN / (FP + TN)
                BAC = (SEN + SPE) / 2
        if epoch % print_freq == 0:
            print(f'Best val Acc: {best_acc:4f}')
            print('-' * 20)
    # 输出最后结果
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    # 保存模型训练的最高准确率
    acc = max(acc_list)
    print(f'Best Acc:, {acc:4f}')
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, TP, TN, FN, FP, outputs, ACC, SEN, SPE, BAC


cfg1 = get_config('config/config1.yaml')
cfg2 = get_config('config/config2.yaml')
cfg3 = get_config('config/config3.yaml')
cfg4 = get_config('config/config4.yaml')
# initialize data
data_dir1 = cfg1['dataset']
data_dir2 = cfg2['dataset']
data_dir3 = cfg3['dataset']
data_dir4 = cfg4['dataset']

K_list = cfg1['K_neigs']
print(K_list)
for K in K_list:
    since = time.time()
    best_train_acc_list = []
    best_val_acc_list = []
    acc = []
    sen = []
    spe = []
    pre = []
    bac = []
    Auc = []
    f1 = []
    print(f'K为： {K:d}', file=mylog)

    # 返回全部数据，标签，还有训练集和测试集的序号和关联矩阵H
    fts1, lbls, idx_train, idx_test, H1 = \
        load_feature_construct_H(data_dir1,
                                 cfg1['subjectnum'],
                                 cfg1['featurenum'],
                                 cfg1['positiveclass'],
                                 cfg1['negativeclass'],
                                 cfg1['m_prob'],
                                 K,
                                 cfg1['is_probH'],
                                 )
    fts1 = fts1.reshape(cfg1['subjectnum'], -1)
    fts2, _, _, _, H2 = \
        load_feature_construct_H(data_dir2,
                                 subjectnum=cfg1['subjectnum'],
                                 featurenum=cfg1['featurenum'],
                                 positiveclass=cfg1['positiveclass'],
                                 negativeclass=cfg1['negativeclass'],
                                 m_prob=cfg2['m_prob'],
                                 K_neigs=K,
                                 is_probH=cfg2['is_probH'],
                                 )
    fts2 = fts2.reshape(cfg1['subjectnum'], -1)
    fts3, _, _, _, H3 = \
        load_feature_construct_H(data_dir3,
                                 subjectnum=cfg1['subjectnum'],
                                 featurenum=cfg1['featurenum'],
                                 positiveclass=cfg1['positiveclass'],
                                 negativeclass=cfg1['negativeclass'],
                                 m_prob=cfg3['m_prob'],
                                 K_neigs=K,
                                 is_probH=cfg3['is_probH'],
                                 )
    fts3 = fts3.reshape(cfg1['subjectnum'], -1)
    fts4, _, _, _, H4 = \
        load_feature_construct_H(data_dir4,
                                 subjectnum=cfg1['subjectnum'],
                                 featurenum=cfg1['featurenum'],
                                 positiveclass=cfg1['positiveclass'],
                                 negativeclass=cfg1['negativeclass'],
                                 m_prob=cfg4['m_prob'],
                                 K_neigs=K,
                                 is_probH=cfg4['is_probH'],
                                 )
    fts4 = fts4.reshape(cfg1['subjectnum'], -1)


    H1 = H1.astype('float')
    H2 = H2.astype('float')
    H3 = H3.astype('float')
    H4 = H4.astype('float')
    H = (H1 * 0.1) + (H2 * 0.2) + (H3 * 0.3) + (H4 * 0.4)


    W = weight.set_weight(H, feat)
    G = utils.hypergraph_utils.generate_G_from_H(H, W, variable_weight=False)
    n_class = int(lbls.max()) + 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ############
    ###########


    # transform data to device放到cpu或者gpu上运行
    feat = torch.Tensor(feat).to(device)
    lbls = torch.Tensor(lbls).squeeze().long().to(device)
    G = torch.Tensor(G).to(device)

    for i in range(cfg1['iteration']):
        shuffle_idx = np.array(range(0, cfg1['subjectnum']))  # 创建一个数组，计数从 0开始。计数到 stop 结束，不包含stop。
        # 作用就是重新排序返回一个随机序列作用类似洗牌
        np.random.shuffle(shuffle_idx)
        n = int(lbls.shape[0] * 0.8)  # 划分训练集和测试集
        idx_train = shuffle_idx[0:n]
        idx_test = shuffle_idx[n:lbls.shape[0]]
        idx_train = torch.Tensor(idx_train).long().to(device)
        idx_test = torch.Tensor(idx_test).long().to(device)
        loss_list = []
        acc_list = []

        # 前⾯加f表⽰格式化字符串，加f后可以在字符串⾥⾯使⽤⽤花括号括起来的变量和表达式，如果字符串⾥⾯没有表达式，那么前⾯加不加f输出应该都⼀样
        print(f"Classification on {cfg1['on_dataset']} dataset!!! class number: {n_class}")

        model_ft = HGNN(in_ch=feat.shape[1],
                        n_class=n_class,
                        n_hid=cfg1['n_hid'],
                        dropout=cfg1['drop_out'])
        model_ft = model_ft.to(device)
        # optimizer = optim.SGD(model_ft.parameters(), lr=cfg1['lr'])
        optimizer = optim.Adam(model_ft.parameters(), lr=cfg1['lr'], weight_decay=cfg1['weight_decay'])
        schedular = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg1['gamma'])
        # criterion 是损失函数的选取，CrossEntropyLoss()是交叉熵损失
        criterion = torch.nn.CrossEntropyLoss()

        print(f'下边是K为{K}的结果')
        _, TP, TN, FN, FP, preds, ACC, SEN, SPE, BAC = train_model(model_ft, criterion, optimizer, schedular,
                                                                          cfg1['max_epoch'],
                                                                          print_freq=cfg1['print_freq'])

        time_elapsed = time.time() - since
        print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        #########画图
        x = range(cfg1['max_epoch'])
        plt.figure(num=1)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(x, loss_list)
        plt.figure(num=2)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.plot(x, acc_list)

        # nn.Module模块中的state_dict变量存放训练过程中需要学习的权重和偏执系数.
        hgw1 = model_ft.state_dict()['hgc1.weight']
        hgw2 = model_ft.state_dict()['hgc2.weight']
        # 需要修改
        feat_score,feat_idx = feature_ex.feature_extraction(hgw1,hgw2,cfg1['featurenum'])
        print(feat_score)
        print(feat_idx)
        import torch.nn.functional as F
        from sklearn.metrics import roc_curve, auc

        ## 画ROC曲线
        y_test = lbls[idx_test].numpy()

        y_score = F.softmax(preds, 1)
        y_score = y_score.detach().numpy()
        y_scores = y_score[0:y_score.shape[0], 1]
        y_scores1 = y_scores[idx_test]
        # for i in range(y_score.shape[0]):
        #     if y_test[i] == 0:
        #         y_scores[i] = y_score[i, 0]
        #     else:
        #         y_scores[i] = y_score[i ,1]
        fpr, tpr, thr = roc_curve(y_test, y_scores1)

        roc_auc = auc(fpr, tpr)

        lw = 2
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        plt.show()
mylog.close()
