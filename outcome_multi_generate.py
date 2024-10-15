import os
import random

import torch
import math
import pandas as pd
import numpy as np
from configs import *
from tqdm import tqdm
from makeData import make_train_data, opti_pdbs
# from models.model4 import PbstNet
from models.model_nonh2_fasta2 import PbstNet
from utils import count_pockets_label_no_water
from biopandas.pdb import PandasPdb
from torchmetrics import Recall, Precision, ConfusionMatrix, F1Score, Accuracy, MatthewsCorrCoef, AUROC
import warnings
warnings.filterwarnings('ignore')
# 计算auroc
aurocMetric = AUROC(task='binary', num_classes=2)
# 计算ACC
acc_mtx = Accuracy(task='binary', num_classes=2)
# 计算矩阵
confus_mtr = ConfusionMatrix(task='binary', num_classes=2)
# 计算精确度--根据task的不同，需要使用不同的参数
precisionMetric = Precision(task='binary', num_classes=2)
# 计算召回率
recall_mex = Recall(task='binary', num_classes=2)
# 计算F1
f1_mtx = F1Score(task='binary', num_classes=2)


def cat_outcome_rank(predicts_sig, labes, Metric, is_iou):
    y_true = labes.cpu().detach()
    # TODO 按照阈值的方法
    y_pred_03 = torch.where(predicts_sig > 0.3, torch.ones_like(predicts_sig), torch.zeros_like(predicts_sig))
    y_pred_03 = y_pred_03.cpu().detach()
    y_pred_05 = torch.where(predicts_sig > 0.5, torch.ones_like(predicts_sig), torch.zeros_like(predicts_sig))
    y_pred_05 = y_pred_05.cpu().detach()
    y_pred_07 = torch.where(predicts_sig > 0.7, torch.ones_like(predicts_sig), torch.zeros_like(predicts_sig))
    y_pred_07 = y_pred_07.cpu().detach()
    y_pred_09 = torch.where(predicts_sig > 0.9, torch.ones_like(predicts_sig), torch.zeros_like(predicts_sig))
    y_pred_09 = y_pred_09.cpu().detach()

    auroc03 = Metric(y_pred_03, y_true)
    auroc05 = Metric(y_pred_05, y_true)
    auroc07 = Metric(y_pred_07, y_true)
    auroc09 = Metric(y_pred_09, y_true)

    # TODO 按照排名的方法
    predicts_signp = predicts_sig.squeeze(1).cpu().detach()
    y_predict_rank = np.argsort(predicts_signp)
    # 创建一个与数组相同长度的全零数组
    result4 = result6 = result8 = result10 = result12 = np.zeros_like(predicts_signp)
    # 计算百分比
    percents4 = math.ceil(result4.shape[0] * 4 / 100)
    percents6 = math.ceil(result6.shape[0] * 6 / 100)
    percents8 = math.ceil(result8.shape[0] * 8 / 100)
    percents10 = math.ceil(result10.shape[0] * 10 / 100)
    percents12 = math.ceil(result12.shape[0] * 12 / 100)

    # 将排名前百分比的值设为1，其余为0
    result4[y_predict_rank[-percents4:]] = 1
    result4_t = torch.tensor(result4).unsqueeze(1)
    auroc_r_4 = Metric(result4_t, y_true)
    # 将排名前百分比的值设为1，其余为0
    result6[y_predict_rank[-percents6:]] = 1
    result6_t = torch.tensor(result6).unsqueeze(1)
    auroc_r_6 = Metric(result6_t, y_true)
    # 将排名前百分比的值设为1，其余为0
    result8[y_predict_rank[-percents8:]] = 1
    result8_t = torch.tensor(result8).unsqueeze(1)
    auroc_r_8 = Metric(result8_t, y_true)
    # 将排名前百分比的值设为1，其余为0
    result10[y_predict_rank[-percents10:]] = 1
    result10_t = torch.tensor(result10).unsqueeze(1)
    auroc_r_10 = Metric(result10_t, y_true)
    # 将排名前百分比的值设为1，其余为0
    result12[y_predict_rank[-percents12:]] = 1
    result12_t = torch.tensor(result12).unsqueeze(1)
    auroc_r_12 = Metric(result12_t, y_true)

    if is_iou == False:
        return [auroc03, auroc05, auroc07, auroc09, auroc_r_4, auroc_r_6, auroc_r_8, auroc_r_10, auroc_r_12]
    else:
        auroc03 = auroc03[1, 1] / (auroc03[1, 1] + auroc03[0, 1] + auroc03[1, 0] + 1e-3)
        auroc05 = auroc05[1, 1] / (auroc05[1, 1] + auroc05[0, 1] + auroc05[1, 0] + 1e-3)
        auroc07 = auroc07[1, 1] / (auroc07[1, 1] + auroc07[0, 1] + auroc07[1, 0] + 1e-3)
        auroc09 = auroc09[1, 1] / (auroc09[1, 1] + auroc09[0, 1] + auroc09[1, 0] + 1e-3)
        auroc_r_4 = auroc_r_4[1, 1] / (auroc_r_4[1, 1] + auroc_r_4[0, 1] + auroc_r_4[1, 0] + 1e-3)
        auroc_r_6 = auroc_r_6[1, 1] / (auroc_r_6[1, 1] + auroc_r_6[0, 1] + auroc_r_6[1, 0] + 1e-3)
        auroc_r_8 = auroc_r_8[1, 1] / (auroc_r_8[1, 1] + auroc_r_8[0, 1] + auroc_r_8[1, 0] + 1e-3)
        auroc_r_10 = auroc_r_10[1, 1] / (auroc_r_10[1, 1] + auroc_r_10[0, 1] + auroc_r_10[1, 0] + 1e-3)
        auroc_r_12 = auroc_r_12[1, 1] / (auroc_r_12[1, 1] + auroc_r_12[0, 1] + auroc_r_12[1, 0] + 1e-3)
        return [auroc03, auroc05, auroc07, auroc09, auroc_r_4, auroc_r_6, auroc_r_8, auroc_r_10, auroc_r_12]


def gen_csv_outs(outs_list, oridf):
    # 将 Tensor 列表转换为 NumPy 数组
    numpy_array = [outs.numpy() for outs in outs_list]
    # 创建 DataFrame，并为列起一个名字叫 names
    one_df = pd.DataFrame(numpy_array, columns=[pdb_code])
    # 获取最大值
    one_df_max = pd.DataFrame(one_df.max(axis=0).values, columns=[pdb_code])
    one_df = pd.concat([one_df, one_df_max], axis=0, ignore_index=True)
    # 合并所有
    all_df = pd.concat([oridf, one_df], axis=1)
    return all_df

if __name__ == '__main__':
    # 设定配体
    ligand_df = pd.read_csv('./data_list/ligand_het.csv')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cuda'
    # 创建网络
    models = PbstNet(em_dim=em_dim, fasta_em_dim=fasta_dim, local_dims=localdims, heads=heads, drops=drops, attnLayerNums=attnLayersNum).to(device)
    # TODO 设定模型
    # ligand_df = pd.read_csv('./data_list/ligand_het.csv')
    load_model = torch.load('./self_model_nonh2_lr00008l0652_continue3/model_file/model_parameter_6.pkl')
    #load_model = torch.load('./model4_all_continue_withh/model_file/model_parameter_4.pkl')
    #load_model = torch.load('./model4_all_continue_h_new6/model_file/model_parameter_2.pkl')
    models.load_state_dict(load_model['model_state_dict'])
    models.eval()

    # TODO 1、设置文件夹位置，输入和输出
    valid_foders = '/home/jianping/chenPDBs/pdbbind_valid'
    out_comes_fd = './out_coms/pdbbind_valid_nonhendMy_20240602/'


    code_list = os.listdir(valid_foders)
    # code_list = random.sample(code_list1, 10)
    all_df_miou = pd.DataFrame()
    all_df_auc = pd.DataFrame()
    all_df_acc = pd.DataFrame()
    all_df_precision = pd.DataFrame()
    all_df_recall = pd.DataFrame()
    all_df_f1 = pd.DataFrame()
    for pdb_code in tqdm(code_list):
        pdb_code_path = os.path.join(valid_foders, pdb_code, pdb_code + '.pdb')
        df_protein = PandasPdb().read_pdb(pdb_code_path)._df
        protein_df, hetatm_data = opti_pdbs(pdb_code_path, ligand_df)
        if protein_df is None:
            print(pdb_code)
            continue

        with torch.no_grad():
            try:
                xyz, atom_index, grouped_xyz, fasta_index, align_list = make_train_data(protein_df, device)
                label = count_pockets_label_no_water(protein_df, hetatm_data, 4.5, device, 'common')  # (F, 1)
                # 6A not good
                #label = count_pockets_label_no_water(protein_df, hetatm_data, 6, device, 'sample')  # (F, 1)
                predict = models(xyz, atom_index, grouped_xyz, fasta_index, align_list)
            except:
                print(pdb_code)
                continue
            # sigmoid 转概率
            predict_sig = torch.sigmoid(predict)
            # TODO 修改这里调用不同的计算方式--iou需要设置为True，传入矩阵
            # IOU
            out_list_miou = cat_outcome_rank(predict_sig, label, confus_mtr, True)
            all_df_miou = gen_csv_outs(out_list_miou, all_df_miou)
            # AUC
            out_list_auc = cat_outcome_rank(predict_sig, label, aurocMetric, False)
            all_df_auc = gen_csv_outs(out_list_auc, all_df_auc)
            # ACC
            out_list_acc = cat_outcome_rank(predict_sig, label, acc_mtx, False)
            all_df_acc = gen_csv_outs(out_list_acc, all_df_acc)
            # Precision
            out_list_precision = cat_outcome_rank(predict_sig, label, precisionMetric, False)
            all_df_precision = gen_csv_outs(out_list_precision, all_df_precision)
            # Recall
            out_list_recall = cat_outcome_rank(predict_sig, label, recall_mex, False)
            all_df_recall = gen_csv_outs(out_list_recall, all_df_recall)
            # F1
            out_list_f1 = cat_outcome_rank(predict_sig, label, f1_mtx, False)
            all_df_f1 = gen_csv_outs(out_list_f1, all_df_f1)

    os.makedirs(out_comes_fd, exist_ok=True)
    all_df_miou.to_csv(out_comes_fd+"iou"+".csv", index=False)
    all_df_auc.to_csv(out_comes_fd+"auc"+".csv", index=False)
    all_df_acc.to_csv(out_comes_fd + "acc" + ".csv", index=False)
    all_df_precision.to_csv(out_comes_fd + "precision" + ".csv", index=False)
    all_df_recall.to_csv(out_comes_fd + "recall" + ".csv", index=False)
    all_df_f1.to_csv(out_comes_fd + "f1" + ".csv", index=False)
    # # # TODO 结果输出文件
    # predict_sig = torch.sigmoid(predict)
    # predict_df = pd.DataFrame(predict_sig.detach().numpy())
    # protein_df['b_factor'] = predict_df
    # ppdb = PandasPdb()
    # ppdb.df['ATOM'] = protein_df
    # ppdb.to_pdb('./2F9W_A_model4.pdb')