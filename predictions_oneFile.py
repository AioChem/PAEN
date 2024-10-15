import os
import sys
import time
import numpy as np
import pandas as pd
import random
import torch
from configs import *
from makeData import opti_pdbs_nonHet, make_train_data
from model_nonh2_fasta2 import PbstNet
from utils import outcome_valid_sigmoid, count_pockets_label_no_water
from biopandas.pdb import PandasPdb
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cuda'
    # 创建网络
    models = PbstNet(em_dim=em_dim, fasta_em_dim=fasta_dim, local_dims=localdims, heads=heads, drops=drops, attnLayerNums=attnLayersNum).to(device)
    # TODO 设定模型参数
    load_model = torch.load('./modelsPTH/self-model-celoss25-lr0001-continue4/model_file/model_parameter_5.pkl')
    models.load_state_dict(load_model['model_state_dict'])
    # torch.save(models, './complete_model.pth')

    # TODO 设定预测文件
    a = '1IG3_A'
    # pdb_file = './pdb_datas/tests_sets/7KHT/7KHT.pdb'
    pdb_file = './pdb_datas/tests_sets/'+a+'/'+a+'.pdb'
    #out_file = './pdb_datas/tests_sets/7KHT/7KHT_predict.pdb'
    out_file = './pdb_datas/tests_sets/'+a+'/'+a+'_predict.pdb'
    #df_protein = PandasPdb().read_pdb(pdb_file)._df
    protein_data = opti_pdbs_nonHet(pdb_file)
    models.eval()
    with torch.no_grad():
        xyz, atom_index, grouped_xyz, fasta_index, align_list = make_train_data(protein_data, device)

        predict = models(xyz, atom_index, grouped_xyz, fasta_index, align_list)
        # # TODO 结果输出文件
        predict_sig = torch.sigmoid(predict)
        predict_df = pd.DataFrame(predict_sig.cpu().detach().numpy())
        protein_data['b_factor'] = predict_df
        #protein_data['b_factor'] = (predict_df > 0.5).astype(int)
        ppdb = PandasPdb()
        ppdb.df['ATOM'] = protein_data
        ppdb.to_pdb(out_file)
