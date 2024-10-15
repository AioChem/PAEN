import os
import sys
import time
import numpy as np
import pandas as pd
import random
import shutil
import torch
from torch import optim
from torch import nn
from configs import *
from tqdm import tqdm
from makeData import make_train_data, opti_pdbs
from model_nonh2_fasta2 import PbstNet
# from models.model4 import PbstNet
from bfocalloss import BinaryFocalLoss
from utils import outcome_valid_sigmoid, count_pockets_label_no_water
# from biopandas.pdb import PandasPdb
import warnings
warnings.filterwarnings('ignore')


def make_files():
    # 创建相应文件夹
    name = input("请输入要保存结果的文件夹(ps: ./results ): ")
    if os.path.exists(name):
        print('文件夹已存在，请重新选择文件夹，程序已结束！')
        sys.exit()
    else:
        model_file_name = os.path.join(name, 'model_file')
        os.makedirs(model_file_name)

        loss_file_name = os.path.join(name, 'loss_file')
        os.makedirs(loss_file_name)

        miou_file_name = os.path.join(name, 'miou_file')
        os.makedirs(miou_file_name)

        precision_file_name = os.path.join(name, 'precision_file')
        os.makedirs(precision_file_name)

        recall_file_name = os.path.join(name, 'recall_file')
        os.makedirs(recall_file_name)

        f1_file_name = os.path.join(name, 'f1_file')
        os.makedirs(f1_file_name)

        mcc_file_name = os.path.join(name, 'mcc_file')
        os.makedirs(mcc_file_name)
    print('对应文件夹创建成功')
    return name


def make_model(lossinfo, is_continue):
    """
    模型参数在这里设置
    :return:
    """
    device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device1 = 'cpu'

    # 创建网络
    models1 = PbstNet(em_dim=em_dim, fasta_em_dim=fasta_dim, local_dims=localdims, heads=heads, drops=drops, attnLayerNums=attnLayersNum).to(device1)
    # 创建优化器
    optimizer1 = optim.SGD(models1.parameters(), lr=learning_rate, momentum=0.9)

    if is_continue:
        load_model = torch.load(continue_models)
        models1.load_state_dict(load_model['model_state_dict'])
        optimizer1.load_state_dict(load_model['optimizer_state_dict'])

    if lossinfo == 'ce-loss':
        weights = torch.tensor([2.5]).to(device1)
        bfloss1 = nn.BCEWithLogitsLoss(pos_weight=weights)
    elif lossinfo == 'focal-loss':
        bfloss1 = BinaryFocalLoss(alpha=0.75, gamma=2)
    else:
        raise ValueError('loss错误ce|focal')
    print('训练参数：', device1, lossinfo, is_continue)
    return device1, models1, bfloss1, optimizer1


def set_codes():
    """
    文件夹信息在这里
    :return:
    """
    # train_foders = '/home/jianping/chenPDBs/all_data'
    train_foders = './pdb_datas/tests_sets'
    all_codes = os.listdir(train_foders)

    # valid_foders = '/home/jianping/chenPDBs/pdbbind_valid'
    valid_foders = './pdb_datas/tests_sets'
    valid_codes = os.listdir(valid_foders)

    #valid_foders = '/home/jianping/chenPDBs/COACH420'
    #valid_codes = os.listdir(valid_foders)

    train_codes = list(set(all_codes) - set(valid_codes))
    # train_codes = random.sample(train_codes, 10)
    # valid_codes = random.sample(valid_codes, 10)
    return train_codes, valid_codes, train_foders, valid_foders


if __name__ == '__main__':
    # 创建文件夹
    name = make_files()
    # 制作模型
    device, models, bfloss, optimizer = make_model(lossinfo='ce-loss', is_continue=is_continues_cof)
    #device, models, bfloss, optimizer = make_model(lossinfo='ce-loss', is_continue=is_continues_cof)

    # 继续训练的话，把这个解开，并且在Config中设定继续的模型位置

    # 定义训练测试文件夹
    train_code, valid_code, train_foder, valid_foder = set_codes()
    # 设定配体
    ligand_df = pd.read_csv('./data_list/ligand_het.csv')
    # 开始训练
    for epoch in range(epochs):

        startTime = time.time()
        # 每个epoch开始的时候设定为训练模式
        models.train()
        TrainLoss = []

        for pdb_code in tqdm(train_code):
            pdb_file = os.path.join(train_foder, pdb_code, pdb_code + '.pdb')
            protein_data, hetatm_data = opti_pdbs(pdb_file, ligand_df)
            label = count_pockets_label_no_water(protein_data, hetatm_data, 4.5, device, 'common')  # (F, 1)
            xyz, atom_index, grouped_xyz, fasta_index, align_list = make_train_data(protein_data, device)
            predict = models(xyz, atom_index, grouped_xyz, fasta_index, align_list)
            train_loss = bfloss(predict, label)

            # 记录损失
            running_loss = train_loss.item()
            TrainLoss.append(running_loss)
            # 回传
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # 保存模型
        Loss_train_t = torch.tensor(TrainLoss)
        torch.save({
            'epoch': epoch,
            'model_state_dict': models.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': Loss_train_t
        }, os.path.join(name, 'model_file', 'model_parameter_{}.pkl'.format(epoch)))
        print('Train_loss:', np.mean(TrainLoss))
        # print(faildcode)
        # 开始验证
        ValidLoss = []
        Miou_list = []
        Recall_list = []
        Precision_list = []
        f1_list = []
        Acc_list = []
        Mcc_list = []
        AUROC_list = []
        # 设为评估
        models.eval()
        with torch.no_grad():
            print('开始评估')
            for pdb_code in tqdm(valid_code):
                pdb_file = os.path.join(valid_foder, pdb_code, pdb_code + '.pdb')
                protein_data, hetatm_data = opti_pdbs(pdb_file, ligand_df)


                label = count_pockets_label_no_water(protein_data, hetatm_data, 4.5, device, 'common')  # (F, 1)
                xyz, atom_index, grouped_xyz, fasta_index, align_list = make_train_data(protein_data, device)
                # 开始预测
                predict = models(xyz, atom_index, grouped_xyz, fasta_index, align_list)
                valid_loss = bfloss(predict, label)


                valid_loss_item = valid_loss.item()
                ValidLoss.append(valid_loss_item)

                # 记录其他的值
                miou, precision, recall, f1, acc, mcc, auroc = outcome_valid_sigmoid(predict, label, confidence_level)
                Miou_list.append(miou)
                Precision_list.append(precision)
                Recall_list.append(recall)
                f1_list.append(f1)
                Acc_list.append(acc)
                Mcc_list.append(mcc)
                AUROC_list.append(auroc)
            print('miou:', np.mean(Miou_list))
            print('precision:', np.mean(Precision_list))
            print('recall:', np.mean(Recall_list))
            print('acc:', np.mean(Acc_list))
            # print('mcc:', np.mean(Mcc_list))
            print('auroc', np.mean(AUROC_list))
            # 保存参数
            validliss = torch.tensor(ValidLoss)
            Miou = torch.tensor(Miou_list)
            precision = torch.tensor(Precision_list)
            Recall0 = torch.tensor(Recall_list)
            f1_t = torch.tensor(f1_list)
            mcc_s = torch.tensor(Mcc_list)

            torch.save(validliss, os.path.join(name, 'loss_file', 'loss_epoch_{}'.format(epoch)))
            torch.save(Miou, os.path.join(name, 'miou_file', 'miou_epoch_{}'.format(epoch)))
            torch.save(Recall0, os.path.join(name, 'recall_file', 'recall_epoch_{}'.format(epoch)))
            torch.save(precision, os.path.join(name, 'precision_file', 'precision_epoch_{}'.format(epoch)))
            torch.save(f1_t, os.path.join(name, 'f1_file', 'f1_score_epoch_{}'.format(epoch)))
            torch.save(mcc_s, os.path.join(name, 'mcc_file', 'mcc_score_epoch_{}'.format(epoch)))