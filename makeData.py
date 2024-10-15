from configs import *
from utils import *
import os
from biopandas.pdb import PandasPdb
import warnings
warnings.filterwarnings('ignore')
cofactor_l43 = ['SO4','SO3','SO2','CO3','OH','PEO','URE','TOU','XL3',
                'DMS','GAI','PO4', 'PO3','EDO', 'POL','VXA','ACT','FMT',
                'NO3','FES','MLI','OXY','NH2','NH4','BME',
                'IPA','THF','COM','COR','PHS','ATO','ACY','ACM',
                'ACN','0HQ','DCE','BCT','EOH','3QD','NO','NO2','3CN','GOL']
lipids_l4 = ['PLM', '11Z', '16Y', 'D0G', 'J1W', 'OHA', 'X90', 'KV6', 'RF8', 'STE', '7NR', 'EOD', 'EW8', 'KTC',
          'WAD', 'XPM', '14U', '87O', 'DCR', 'VGJ', '87K', 'EKG', 'VLP', 'ZP7', '14V', 'EO3', '2JT', '3XQ',
          'F23', 'BZV', 'G2A', 'NKO', 'PNR', '7PO', '16E', '9R2', 'VVA', 'LPX', 'PVC', 'ER0', '82T', 'PGM',
          'K6G', 'KIP', 'LP3', 'WJS', 'T80', 'U9V', '6IY', 'PXS', '1TF', '4AG', '9PE', 'F57', 'Z41', 'SBJ',
          '1T1', '6UL', '9XX', 'E2V', 'MYY', 'YZY', 'Z0P', '88I', '8ND', 'EL6', 'U9N', 'CUY', 'DGA', 'L44',
          'LPP', 'PX6', 'WO9', 'CQI', 'P3X', 'PDJ', 'D21', 'E8Q', '0V9', '8PE', 'M7U', 'PEF', 'PWE', '3PH',
          'HGP', 'HGX', 'MGE', 'PX8', 'X81', '6OU', '6V6', '9Y0', 'DKB', 'HYC', 'LHG', 'PEV', '0W3', '70E',
          'DGG', 'GOT', 'H3T', 'KUT', 'PCF', 'PTY', '3PE', 'D3D', 'IKV', 'L9Q', 'LTV', 'P50', 'PEE', 'PEW',
          'PGT', 'PGV', 'PGW', 'YFP', '6PL', 'BJR', 'CPL', 'D39', 'K9G', 'LBN', 'PC7', 'POV', 'PSC', 'Q3G',
          '1O2', '3TF', 'MW9', 'PEK', 'PGK', 'L9R', 'M2R', 'P5S', 'PC1', 'PC9', 'PS2', 'SQD', '85R', 'LMG',
          'MVV', 'P5L', 'PC6', 'PO9', 'UYH', 'PII', 'SRV', '4RF', '8IO', 'OQ5', 'PIE', 'YBG', '9YF', 'EV9',
          'P3A', 'PCK', '8IJ', 'B7N', 'YPC', '1K1', 'PLD', 'EIJ', 'T7X', 'PCJ', 'IG7', 'PIK', 'TGL', 'T7M',
          '2Y5', 'DGD', 'PIZ', 'IEP', 'KXP', 'PT5', 'T8X', '06O', 'GMM', '0HJ', 'XPX', '5PL', 'OJF', 'IX7',
          'CDL', 'K8U', 'PRD_002432', 'TQN', 'ELA', 'OLA', '14Y', 'RCL', 'T25', '11O', 'A6L', 'OLB', 'OLC',
          'YOG', 'NKP', 'OU9', 'AQR', 'QZQ', 'CBW', '42H', 'LSC', 'S12', 'AF7', '7Q9', 'GR3', 'CBO', 'VCG',
          'LOP', 'FJL', '2OB', 'MX7', 'OZ2', 'TAJ', 'P0E', 'RXY', 'DR9', 'GR4', 'U3G', '17F', 'PCW', 'WSS',
          '1L2', 'U3D', 'P8X', 'SH0', '0UK', 'COJ', '8F5', 'GGD', '58A', 'HZL', 'WES', 'SMW', 'QSR', 'C9V',
          'CLR', '0GV', '5JK', '94R', 'CO1', 'HC2', 'HC3', 'HC9', 'HCD', 'HCR', '2DC', 'KJX', 'YUV', '0T9',
          'L39', 'C3S', '1KG', 'ECK', 'Y01', 'DU0', '9Z9', 'CLL', 'HOB', 'HOL', 'YUY', 'I7Y', 'J4U', 'Q7G',
          'YJ0', 'RET', '7AB', '9CR', 'REA', 'ETR', 'FEN']

def opti_pdbs(pdb_files, ligand_dfs):
    """
    处理文件，转换为可用的Df数据
    :param pdb_files:
    :param ligand_dfs:
    :return:
    """
    if not os.path.exists(pdb_files):
        return None, None
    df_proteins = PandasPdb().read_pdb(pdb_files)._df  # 读取文件
    protein11 = df_proteins['ATOM']
    # 去掉氢原子进行训练
    protein2 = protein11[protein11['element_symbol'] != 'H']
    protein2 = protein2.reset_index(drop=True)
    if len(protein2) > max_len:
        return None, None
    #if len(protein2) < 500:
        #return None, None
    hetatms = df_proteins['HETATM']
    # 仅处理小分子配体
    hetatm_optis = hetatms[hetatms['residue_name'].isin(ligand_dfs['het'])]
    hetatm_optis = hetatm_optis[~hetatm_optis['residue_name'].isin(cofactor_l43)]
    # hetatm_optis = hetatm_optis[~hetatm_optis['residue_name'].isin(lipids_l4)]
    # a = len(hetatm_optis)
    # 配体少的不要
    if len(hetatm_optis) < 3:
        return None, None
    # data_dfs = pd.concat([protein2, hetatm_optis], axis=0)
    # 修补文件数据
    protein2['insertion'] = protein2['insertion'].fillna('')
    protein2 = protein2[protein2['insertion'] == '']
    protein2 = protein2.reset_index(drop=True)
    # 覆盖name和number
    protein2['news_rsd'] = protein2['residue_name'].astype(str) + protein2['residue_number'].astype(str)

    return protein2, hetatm_optis


def opti_pdbs_nonHet(pdb_files):
    """
    处理文件，转换为可用的Df数据
    :param pdb_files:
    :param ligand_dfs:
    :return:
    """
    if not os.path.exists(pdb_files):
        return None, None
    df_proteins = PandasPdb().read_pdb(pdb_files)._df  # 读取文件
    protein11 = df_proteins['ATOM']

    has_loc_atom = protein11['alt_loc'].str.contains('.+')
    if sum(has_loc_atom) > 0:
        atom_df_dup = protein11[has_loc_atom]
        # 取出不是A的行
        atom_df_dup = atom_df_dup[atom_df_dup['alt_loc'] != 'A']
        # 拼接到后面
        df_atom = pd.concat([protein11, atom_df_dup], axis=0)
        # 去掉这些行，不保留结果，这样就留下A了
        protein11 = df_atom.drop_duplicates(keep=False)

    # 去掉氢原子进行训练
    protein2 = protein11[protein11['element_symbol'] != 'H']
    protein2 = protein2.reset_index(drop=True)

    # data_dfs = pd.concat([protein2, hetatm_optis], axis=0)
    # 修补文件数据
    protein2['insertion'] = protein2['insertion'].fillna('')
    protein2 = protein2[protein2['insertion'] == '']
    protein2 = protein2.reset_index(drop=True)
    # 覆盖name和number
    protein2['news_rsd'] = protein2['residue_name'].astype(str) + protein2['residue_number'].astype(str)

    return protein2

def opti_pdbs_predict(pdb_files, ligand_dfs):
    """
    处理文件，转换为可用的Df数据
    :param pdb_files:
    :param ligand_dfs:
    :return:
    """
    if not os.path.exists(pdb_files):
        return None, None
    df_proteins = PandasPdb().read_pdb(pdb_files)._df  # 读取文件
    protein2 = df_proteins['ATOM']
    # 去掉氢原子进行训练
    #protein2 = protein11[protein11['element_symbol'] != 'H']
    #protein2 = protein2.reset_index(drop=True)
    if len(protein2) > max_len:
        return None, None
    if len(protein2) < 500:
        return None, None
    hetatms = df_proteins['HETATM']
    # 仅处理小分子配体
    hetatm_optis = hetatms[hetatms['residue_name'].isin(ligand_dfs['het'])]
    # hetatm_optis = hetatm_optis[hetatm_optis['residue_name'] != 'GOL']
    # a = len(hetatm_opti)
    # 配体少的不要
    if len(hetatm_optis) < 3:
        return None, None
    # data_dfs = pd.concat([protein2, hetatm_optis], axis=0)
    # 修补文件数据
    protein2['insertion'] = protein2['insertion'].fillna('')
    protein2 = protein2[protein2['insertion'] == '']
    protein2 = protein2.reset_index(drop=True)
    # 覆盖name和number
    protein2['news_rsd'] = protein2['residue_name'].astype(str) + protein2['residue_number'].astype(str)

    return protein2, hetatm_optis

# def make_train_data(protein_df, het_df,  device, is_fasta, label_ra):
def make_train_data(protein_df, device):
    """
    制作模型需要的数据

    all_data: pandas读取的csv对象
    label: n 1 float32
    """

    # 一、计算扩充列表-每个氨基酸有多少个原子的扩充列表
    count_list = countResidueToatom(protein_df)  # list[n]

    # 二、计算原子的类型标签，对应到编码
    # atom_type = count_atom_type(protein_df, device)  # 66种编码
    atom_type = count_atom_type_lower(protein_df, device)  # 6种编码  (1, n)

    # 三、计算氨基酸序列
    fasta = count_fasta_type(protein_df, device)  # (1, F)

    # 四、计算xyz坐标和范围扩充点的索引
    xyz, ca_index = count_ca_atom(protein_df, list_range, device)  # xyz:(1, n 3)  ca_index:list[[1, F, 8],[1, F, 16],[1, F, 24]]

    # 如果为None是预测，不需要进行标签计算
    # if is_fasta is None:
    return xyz, atom_type, ca_index, fasta, count_list
