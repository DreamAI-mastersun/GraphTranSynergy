import random
import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch.utils.data as Data
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset

from utils_test import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from model.GraphTransformer import GraphTransformerNet
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 引入学习率调度器


# training function at each epoch
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    # train_loader = np.array(train_loader)
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data[0].y.view(-1, 1).long().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = loss_fn(output, y)
        # print('loss', loss)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {}  Loss: {:.6f}'.format(epoch, loss.item()))


def predicting(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output = model(data1, data2)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


# modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]



TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.0001
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
# file = ['breast','colon','lung','melanoma','ovarian','prostate']
# file = ['d00','d11','d22','d33','d44','dd0and5']
file = ['c00','c11','c22','c33','c44']


# CPU or GPU

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

for datafile in file:

    independent_drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1')
    independent_drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2')
    print('independent_drug1_data[0]', independent_drug1_data[0])
    lenth = len(independent_drug1_data)

    # leave_tissue_filename1 = 'leave_' + datafile + '_drug1'
    # leave_tissue_filename2 = 'leave_' + datafile + '_drug2'

    leave_comb_filename1 = 'leave_' + datafile + '_drug1'
    leave_comb_filename2 = 'leave_' + datafile + '_drug2'

    # leave_drug_filename1 = 'leave_' + datafile + '_drug1'
    # leave_drug_filename2 = 'leave_' + datafile + '_drug2'

    drug1_data = TestbedDataset(root='data', dataset=leave_comb_filename1)
    drug2_data = TestbedDataset(root='data', dataset=leave_comb_filename2)
    lenth = len(drug1_data)
    random_num = random.sample(range(0, lenth), lenth)
    drug1_data = drug1_data[random_num]
    drug2_data = drug2_data[random_num]

    # 构建训练集
    drug1_loader_train = DataLoader(drug1_data, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug2_loader_train = DataLoader(drug2_data, batch_size=TRAIN_BATCH_SIZE, shuffle=None)

    # 构建独立测试集
    independent_drug1_loader_test = DataLoader(independent_drug1_data, batch_size=TEST_BATCH_SIZE, shuffle=None)
    independent_drug2_loader_test = DataLoader(independent_drug2_data, batch_size=TEST_BATCH_SIZE, shuffle=None)

    modeling = GraphTransformerNet
    model = modeling().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 初始化调度器
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)


    model_file_name = 'data/result/sun_pre_test/' + datafile + '_GraphTransformer--model.pt'
    file_AUCs = 'data/result/sun_pre_test/' + datafile + '_GraphTransformer--AUCs.txt'
    # model_file_name = 'data/result/sun_pre_test/' + datafile + '_GAT--model.pt'
    # file_AUCs = 'data/result/sun_pre_test/' + datafile + '_GAT--AUCs.txt'
    AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    best_auc = 0

    for epoch in range(NUM_EPOCHS):
        train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)

        if (epoch+1) % 100 != 0:

            T, S, Y = predicting(model, device, independent_drug1_loader_test, independent_drug2_loader_test)
            # torch.save(model.state_dict(), model_file_name)
            independent_num_file_name = 'data/result/sun_pre_test/' + datafile + '--result' + str(epoch) + '.csv'
            independent_num = []
            # independent_num.append(test_num)
            # independent_num.append(T)
            independent_num.append(Y)
            independent_num.append(S)
            txtDF = pd.DataFrame(data=list(map(list, zip(*independent_num))))
            # txtDF.to_csv(independent_num_file_name, index=False, header=False)

            # T is correct label
            # S is predict score
            # Y is predict label

            # compute preformence
            # print('============')
            # print(T.shape)
            # print(S.shape)
            # print('============')
            AUC = roc_auc_score(T, S)
            precision, recall, threshold = metrics.precision_recall_curve(T, S)
            PR_AUC = metrics.auc(recall, precision)
            BACC = balanced_accuracy_score(T, Y)
            tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
            TPR = tp / (tp + fn)
            PREC = precision_score(T, Y)
            ACC = accuracy_score(T, Y)
            KAPPA = cohen_kappa_score(T, Y)

            # scheduler.step(AUC)  # 更新学习率

            # save data
            if best_auc < AUC:
                best_auc = AUC
                print('=======================')
                print("Best AUC:")
                print(best_auc)
                print('=======================')
                torch.save(model.state_dict(), model_file_name)
                independent_num = []
                independent_num.append(T)
                independent_num.append(Y)
                independent_num.append(S)
                txtDF = pd.DataFrame(data=independent_num)
                txtDF.to_csv(independent_num_file_name, index=False, header=False)

                # sns.set()
                # f, ax = plt.subplots()
                # CM = confusion_matrix(T, Y)
                # print(CM)
                # sns.heatmap(CM, annot=True, ax=ax, fmt='.20g')
                # ax.set_title('DeepDDS(GATNet) confusion matrix')  # 标题
                # ax.set_xlabel('Predict')  # x轴
                # ax.set_ylabel('True')  # y轴
                # plt.savefig('D:\GraphDTA-master\data\\result\photo\GATNet_cm.png')

                AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA]
                save_AUCs(AUCs, file_AUCs)
            # torch.save(model.state_dict(), model_file_name)
            print("BEST_AUC", best_auc)