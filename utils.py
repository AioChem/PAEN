import pandas as pd
import numpy as np
import torch
# from biopandas.pdb import PandasPdb
from torchmetrics import Recall, Precision, ConfusionMatrix, F1Score, Accuracy, MatthewsCorrCoef, AUROC
# from configs import confidence_level

aa_list = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP',
           'TYR', 'ASP', 'ASN', 'GLU', 'LYS', 'GLN', 'MET',
           'SER', 'THR', 'CYS', 'PRO', 'HIS', 'ARG', 'X']

#atom_list = ['N', 'C', 'O', 'S', 'H', 'X']
atom_list = ['N', 'C', 'O', 'S', 'X']
std_atom = [
    'CA', 'N', 'C', 'O', 'CB', 'CG', 'CD2', 'CD1', 'CG1', 'CG2', 'CD',
    'OE1', 'OE2', 'OG', 'OG1', 'OD1', 'OD2', 'CE', 'NZ', 'NE', 'CZ',
    'NH2', 'NH1', 'ND2', 'CE2', 'CE1', 'NE2', 'OH', 'ND1', 'SD', 'SG',
    'NE1', 'CE3', 'CZ3', 'CZ2', 'CH2', 'P', "C3'", "C4'", "O3'", "C5'",
    "O5'", "O4'", "C1'", "C2'", "O2'", 'OP1', 'OP2', 'N9', 'N2', 'O6',
    'N7',  'C8', 'N1', 'N3', 'C2', 'C4', 'C6', 'C5', 'N6', 'N4', 'O2',
    'O4', 'OXT', 'UNK', 'X']
chain_list = ['A','B','C','D','E','F','G','H','I','J','K'
              'L','M','N','O','P','Q','R','S','T','U','V'
              'W','X','Y','Z','X']
aa_classes = {}
for idx, aa in enumerate(aa_list):
    aa_classes[aa] = idx
atom_classes = {}
for idx, atom in enumerate(atom_list):
    atom_classes[atom] = idx
stda_classes = {}
for idx, stda in enumerate(std_atom):
    stda_classes[stda] = idx
chain_classes = {}
for idx, cid in enumerate(chain_list):
    chain_classes[cid] = idx

def pc_normalize(pc):
    """
    input：1，n，3
    点云数据标准化
    point-cloud normal
    return: n,3
    """
    if pc.is_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    pc = np.array(pc.squeeze(0).cpu().detach()).astype(float)
    '标准化坐标值'
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    pc = torch.tensor(pc, device=device, dtype=torch.float32)
    return pc


def square_distance(src, dst):
    """
    计算距离
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: 所有的点，source points, [B, N, C]
        dst: 目标点，target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: 查询半径
        nsample: 取样的个数
        xyz: 所有点的集合【batch,num,3】all points, [B, N, 3]
        new_xyz: 需要查询的点【batch,sample,3】query points, [B, S, 3]
    Return:
        group_idx: 查询后的索引【batch,查询点,取样的个数】grouped points index, [B, S, nsample]
        group_idx2: 比如你用10个点，查在100个点内，这10个点每个点附近相邻距离内的5个点，返回的就是(batch,10,5)，是索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def index_points(points, idx):
    """
    根据索引取点
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def countResidueToatom(protein_pd):
    """
    网络内的Align计算方式，把序列的信息扩充给原子，需要提前给出怎么扩充的，这里做一个计算，给出计算列表
    也就是计算当前pdb文件每个氨基酸有多少个原子，组合成一个列表就行
    """
    # 转换nan为空
    #protein_pd['insertion'] = protein_pd['insertion'].fillna('')
    #protein_pd = protein_pd[protein_pd['insertion'] == '']
    # 覆盖1A1B这种氨基酸
    protein_pd['residue_number'] = protein_pd['residue_number'].astype(str)+protein_pd['insertion']

    mask = protein_pd['news_rsd'].ne(protein_pd['news_rsd'].shift())
    # mask = protein_pd['residue_number'].ne(protein_pd['residue_number'].shift())
    groups = mask.cumsum()
    results = protein_pd.groupby(groups).cumcount() + 1
    protein_pd['times'] = results
    protein_pd['count_tmp'] = (protein_pd['times'].diff() != 1).cumsum()
    count_list = protein_pd.groupby('count_tmp')['times'].max().tolist()
    return count_list


def count_atom_type_lower(protein_pd, device):
    """
    查询原子的类型
    """
    atom_type = protein_pd.loc[:, ['element_symbol']]
    # replace not in list to X
    atom_type['element_symbol'] = atom_type['element_symbol'].replace({val: 'X' for val in atom_type['element_symbol'] if val not in atom_list})
    # replace int
    atom_type['element_symbol'] = atom_type['element_symbol'].map(atom_classes)
    atom_type_t = torch.tensor(atom_type.values, device=device, dtype=torch.long)
    # print(1)
    return atom_type_t.transpose(1, 0)


def count_fasta_type(protein_pd, device):
    """
    计算氨基酸序列，对应成数字表示，用于后续wordembding
    :param protein_pd:
    :param device:
    :return:
    """
    # print(1)
    # 在前面的函数覆盖过了，这里不需要再操作了
    # protein_pd['residue_number'] = protein_pd['residue_number'].astype(str)+protein_pd['insertion']
    fasta_type = protein_pd.loc[:, ['residue_name', 'residue_number']]
    mask = fasta_type['residue_number'].ne(fasta_type['residue_number'].shift())
    fasta_results = fasta_type[mask]

    fasta_results = fasta_results.loc[:, ['residue_name']]
    # replace not in aa

    fasta_results['residue_name'] = fasta_results['residue_name'].replace(
        {val: 'X' for val in fasta_results['residue_name'] if val not in aa_list})
    fasta_results['residue_name'] = fasta_results['residue_name'].map(aa_classes)

    fasta_type_t = torch.tensor(fasta_results.values, device=device, dtype=torch.long)
    return fasta_type_t.transpose(1, 0)


def count_ca_atom(protein_pd, list_range, device):
    """
     以阿尔法碳为中心点，查询附近的原子，返回索引
    :param protein_pd: 蛋白的datafram格式
    :param list_range: 取值的范围和点的数量列表：[[4, 5, 6],[16, 32, 64]]
    :param device:
    :return: xyz：标准化后的原子坐标(n,3)
    """
    xyz = protein_pd.loc[:, ['x_coord','y_coord','z_coord']]
    # 重建索引，这一步很重要，否则后续索引会错误
    xyz = xyz.reset_index(drop=True).values
    xyz_t = torch.tensor(xyz, device=device, dtype=torch.float32)
    # 这里转换维度是因为query_ball_point函数要用
    xyz_t = xyz_t.unsqueeze(0)

    ca = protein_pd[protein_pd['atom_name'] == 'CA']
    ca_xyz = ca.loc[:, ['x_coord', 'y_coord', 'z_coord']]
    ca_xyz = ca_xyz.reset_index(drop=True).values

    ca_xyz_t = torch.tensor(ca_xyz, device=device, dtype=torch.float32)
    ca_xyz_t = ca_xyz_t.unsqueeze(0)

    # list_range:  [[4, 5, 6],[16, 32, 64]]
    centroids_all_index = []
    for i in range(len(list_range[0])):
        # 根据中心点，查询原始文件附近的原子
        centroidsindex = query_ball_point(radius=list_range[0][i], nsample=list_range[1][i], xyz=xyz_t, new_xyz=ca_xyz_t)
        # centroids_all_index.append(centroidsindex.squeeze(0))
        centroids_all_index.append(centroidsindex)
    # ca_xyz_centri= pc_normalize_ten(ca_xyz_t)
    # xyz_centri = pc_normalize_ten(xyz_t)
    xyz_centri = pc_normalize(xyz_t)
    #return xyz_centri.unsqueeze(0), centroids_all_index, ca_xyz_centri.unsqueeze(0)
    return xyz_centri.unsqueeze(0), centroids_all_index


def outcome_valid_sigmoid(y_pred, y_true, confidence_level):
    """
    输出结果的计算监测
    y_pred: (n, 1)---且不需要经过sigmoid
    y_true: [n]---一个一维分类列表
    """
    # 查询, 置信度
    y_pred_sig = torch.sigmoid(y_pred)
    # y_pred_sig = y_pred_sig.detach().numpy()
    y_pred_07 = torch.where(y_pred_sig > confidence_level, torch.ones_like(y_pred), torch.zeros_like(y_pred))
    # y_pred_07 = y_pred_07.squeeze(1)

    # cpu
    y_pred = y_pred_07.cpu().detach()
    y_true = y_true.cpu().detach()

    # 计算精确度--根据task的不同，需要使用不同的参数
    precisionMetric = Precision(task='binary', num_classes=2)  # 2分类，看一下分类1的精确度就可以（主要看准不准）
    precision = precisionMetric(y_pred, y_true)
    # 计算召回率
    recall_mex = Recall(task='binary', num_classes=2)
    recall = recall_mex(y_pred, y_true)
    # 计算F1
    f1_mtx = F1Score(task='binary', num_classes=2)
    f1 = f1_mtx(y_pred, y_true)
    # 计算ACC
    acc_mtx = Accuracy(task='binary', num_classes=2)
    acc = acc_mtx(y_pred, y_true)
    # 计算MCC
    mcc_mtx = MatthewsCorrCoef(task='binary', num_classes=2)
    mcc = mcc_mtx(y_pred, y_true)
    # 计算auroc
    aurocMetric = AUROC(task='binary', num_classes=2)
    auroc = aurocMetric(y_pred, y_true)
    # 计算矩阵
    confus_mtr = ConfusionMatrix(task='binary', num_classes=2)
    current = confus_mtr(y_pred, y_true)
    # 计算iou
    Iou = current[1, 1]/(current[1, 1]+current[0, 1]+current[1, 0]+1e-3)
    return Iou, precision, recall, f1, acc, mcc, auroc


def count_pockets_label_no_water(protein_df, het_df, radius, device, isfasta='common'):
    """
    根据配体和附近的水计算口袋，返回分类标签用于计算loss
    :param pdb_files: pdb文件的绝对路径
    :param radius: 配体附近多少半径计算口袋
    :param device:
    :return:
    """
    # dfp = PandasPdb()
    # dfp.read_pdb(pdb_files)
    # 获取全部xyz坐标
    df_atom_xyz = protein_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].values
    # 转tensor
    df_atom_t = torch.Tensor(np.array(df_atom_xyz, dtype=np.float32))
    # 创建和原子数目相同全为False的列表
    pocket_list = torch.zeros(df_atom_t.size()[0], dtype=torch.bool)

    # 得到去水后的配体坐标
    df_rm_w = het_df[het_df['residue_name'] != 'HOH']
    # 重建索引
    df_rm_w = df_rm_w.reset_index(drop=True)
    # 获取配体名称
    # ligand_names = set(df_rm_w["residue_name"])

    # 根据配体的每个原子，查询原始文件内的原子，并且查找交集
    for index in range(len(df_rm_w)):
        # 获取配体单个原子
        het_row = df_rm_w.loc[index, ['x_coord', 'y_coord', 'z_coord']].values
        het_row_t = torch.Tensor(np.array(het_row, dtype=np.float32))
        # 计算距离
        distances = torch.sqrt(torch.sum((df_atom_t - het_row_t) ** 2, dim=1))
        # 判断距离，返回索引
        islist_one = distances < radius
        # 合并只要为True的值
        pocket_list = islist_one | pocket_list
    # 转换成取值TF
    isdf = pd.Series(pocket_list)
    # 合并查询并赋值
    pidf = protein_df[isdf]
    if isfasta == 'sample':
        """
        直接输出点，不扩充到氨基酸链
        """
        merge_df = pd.merge(protein_df, pidf, how='left', on=['x_coord', 'y_coord', 'z_coord'], indicator=True)
        # 都为True的表示有重叠，重叠的数据标为1，反之为0
        merge_df['_merge'] = np.where(merge_df['_merge'] == 'both', 1, 0)
        # 转为标签label
        label_t = torch.tensor(merge_df['_merge'], device=device, dtype=torch.float32)
        return label_t.unsqueeze(1)
    elif isfasta == 'common':
        """
        将结果对应到氨基酸，作为点输出
        """
        # 得到Label的氨基酸
        unique_merged_column = pidf['news_rsd'].drop_duplicates()
        # 查询对应氨基酸
        protein_df['Result'] = protein_df['news_rsd'].isin(unique_merged_column).astype(int)
        label_t = torch.tensor(protein_df['Result'], device=device, dtype=torch.float32)
        return label_t.unsqueeze(1)

    elif isfasta == 'fasta':
        """
        输出氨基酸序列的结果标签
        """
        unique_merged_column = pidf['news_rsd'].drop_duplicates()
        fasta_column = protein_df['news_rsd'].drop_duplicates()
        # 将两个 DataFrame 按照 'MergedColumn' 列进行合并，使用左连接
        merged_df = pd.merge(fasta_column, unique_merged_column, how='left', indicator=True)
        merged_df['_merge'] = np.where(merged_df['_merge'] == 'both', 1, 0)
        label_t = torch.tensor(merged_df['_merge'], device=device, dtype=torch.float32)
        return label_t.unsqueeze(1)


def count_protein_chain(protein_df, device):
    chainid = protein_df.loc[:, ['chain_id']]
    chainid['chain_id'] = chainid['chain_id'].replace({val: 'X' for val in chainid['chain_id'] if val not in chain_list})
    chainid_eb = chainid['chain_id'].map(chain_classes)
    chainid_t = torch.tensor(chainid_eb.values, device=device, dtype=torch.int64)
    chainid_t = chainid_t.unsqueeze(0)
    return chainid_t


def count_label_pdbBind(protein_pd, pockets_pd, device):
    """
    ji suan biao qian
    """
    protein_xyz = protein_pd.loc[:, ['x_coord','y_coord','z_coord']]
    pockets_xyz = pockets_pd.loc[:, ['x_coord', 'y_coord', 'z_coord']]
    merge_df = pd.merge(protein_xyz, pockets_xyz, how='left', on=['x_coord','y_coord','z_coord'], indicator=True)
    merge_df['_merge'] = np.where(merge_df['_merge'] == 'both', 1, 0)
    label_t = torch.tensor(merge_df['_merge'], device=device, dtype=torch.long)
    return label_t

def count_pockets_label(protein_df, het_df, radius, device):
    """
    根据配体和附近的水计算口袋，返回分类标签用于计算loss
    :param pdb_files: pdb文件的绝对路径
    :param radius: 配体附近多少半径计算口袋
    :param device:
    :return:
    """
    # dfp = PandasPdb()
    # dfp.read_pdb(pdb_files)
    # 获取全部xyz坐标
    df_atom_xyz = protein_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].values
    # 转tensor
    df_atom_t = torch.Tensor(np.array(df_atom_xyz, dtype=np.float32))
    # 创建和原子数目相同全为False的列表
    pocket_list = torch.zeros(df_atom_t.size()[0], dtype=torch.bool)

    # 得到去水后的配体坐标
    df_rm_w = het_df[het_df['residue_name'] != 'HOH']
    # 重建索引
    df_rm_w = df_rm_w.reset_index(drop=True)
    # 得到只有水的坐标
    df_w = het_df[het_df['residue_name'] == 'HOH']
    # 重建索引，丢弃原本的索引
    df_w = df_w.reset_index(drop=True)
    # 创建类似大小的全False列表
    water_list = torch.zeros(len(df_w), dtype=torch.bool)
    # 取所有水的xyz坐标
    df_w_xyz = df_w.loc[:, ['x_coord', 'y_coord', 'z_coord']].values
    df_w_t = torch.Tensor(np.array(df_w_xyz, dtype=np.float32))

    # 获取配体名称
    ligand_names = set(df_rm_w["residue_name"])
    # 遍历配体，查询附近的水
    for i in ligand_names:
        for index1 in range(len(df_rm_w)):
            # 获取配体单个原子
            het_row = df_rm_w.loc[index1, ['x_coord', 'y_coord', 'z_coord']].values
            het_row_t = torch.Tensor(np.array(het_row, dtype=np.float32))
            # 计算水的矩阵到原子的距离
            distances_w = torch.sqrt(torch.sum((df_w_t - het_row_t) ** 2, dim=1))
            # 判断距离，返回索引。配体附近1A范围内的水需要保留
            islist_one_water = distances_w < 1
            # 合并只要为True的值
            water_list = islist_one_water | water_list
    # 转换成取值
    water_list_df = pd.Series(water_list)
    # 获取水的信息
    pockets_water_df = df_w[water_list_df]

    # 合并配体和口袋附近的水为一个新的配体datafram
    df_het_water = pd.concat([df_rm_w, pockets_water_df])
    df_het_water = df_het_water.reset_index(drop=True)

    # 根据配体的每个原子，查询原始文件内的原子，并且查找交集
    for index in range(len(df_het_water)):
        # 获取配体单个原子
        het_row = df_het_water.loc[index, ['x_coord', 'y_coord', 'z_coord']].values
        het_row_t = torch.Tensor(np.array(het_row, dtype=np.float32))
        # 计算距离
        distances = torch.sqrt(torch.sum((df_atom_t - het_row_t) ** 2, dim=1))
        # 判断距离，返回索引
        islist_one = distances < radius
        # 合并只要为True的值
        pocket_list = islist_one | pocket_list
    # 转换成取值TF
    isdf = pd.Series(pocket_list)

    # 合并查询并赋值
    pidf = protein_df[isdf]
    merge_df = pd.merge(protein_df, pidf, how='left', on=['x_coord', 'y_coord', 'z_coord'], indicator=True)
    # 都为True的表示有重叠，重叠的数据标为1，反之为0
    merge_df['_merge'] = np.where(merge_df['_merge'] == 'both', 1, 0)

    # print((merge_df['_merge'].count(1)))
    # 转为标签label
    label_t = torch.tensor(merge_df['_merge'], device=device, dtype=torch.float32)
    #c = label_t.sum()
    #print(c)
    # 保存文件
    # dfp.to_pdb(path=out_file)
    #merge_df['_merge'].to_csv('loostest.csv')
    return label_t.unsqueeze(1)


def outcome_valid_sigmoid_sequences(oridf, y_pred, y_true, confidence_level):
    """
    输出结果的计算监测,计算序列的值，但是精度没有很大提升。口袋还是在附近，只不过作用氨基酸变的不太一样了
    y_pred: (n, 1)---且不需要经过sigmoid
    y_true: [n]---一个一维分类列表
    """
    # 查询, 置信度
    y_pred_sig = torch.sigmoid(y_pred)
    y_pred_07 = torch.where(y_pred_sig > confidence_level, torch.ones_like(y_pred), torch.zeros_like(y_pred))
    # cpu
    y_pred = y_pred_07.cpu().detach().numpy()

    # 合并查询并赋值
    pidf = oridf[y_pred == 1]
    unique_merged_column = pidf['news_rsd'].drop_duplicates()
    fasta_column = oridf['news_rsd'].drop_duplicates()
    # 将两个 DataFrame 按照 'MergedColumn' 列进行合并，使用左连接
    merged_df = pd.merge(fasta_column, unique_merged_column, how='left', indicator=True)
    merged_df['_merge'] = np.where(merged_df['_merge'] == 'both', 1, 0)
    label_t = torch.tensor(merged_df['_merge'], device='cpu', dtype=torch.float32).unsqueeze(1)
    y_pred = label_t.cpu().detach()
    y_true = y_true.cpu().detach()

    # 计算精确度--根据task的不同，需要使用不同的参数
    precisionMetric = Precision(task='binary', num_classes=2)  # 2分类，看一下分类1的精确度就可以（主要看准不准）
    precision = precisionMetric(y_pred, y_true)
    # 计算召回率
    recall_mex = Recall(task='binary', num_classes=2)
    recall = recall_mex(y_pred, y_true)
    # 计算F1
    f1_mtx = F1Score(task='binary', num_classes=2)
    f1 = f1_mtx(y_pred, y_true)
    # 计算ACC
    acc_mtx = Accuracy(task='binary', num_classes=2)
    acc = acc_mtx(y_pred, y_true)
    # 计算MCC
    mcc_mtx = MatthewsCorrCoef(task='binary', num_classes=2)
    mcc = mcc_mtx(y_pred, y_true)
    # 计算auroc
    aurocMetric = AUROC(task='binary', num_classes=2)
    auroc = aurocMetric(y_pred, y_true)
    # 计算矩阵
    confus_mtr = ConfusionMatrix(task='binary', num_classes=2)
    current = confus_mtr(y_pred, y_true)
    # 计算iou
    Iou = current[1, 1]/(current[1, 1]+current[0, 1]+current[1, 0]+1e-3)
    return Iou, precision, recall, f1, acc, mcc, auroc


if __name__ == '__main__':
    from makeData import make_train_data
    from biopandas.pdb import PandasPdb
    device = 'cpu'
    pdb_file = '/Users/chen/PBSTNet/pdb_data/pdb1a0f.ent'
    df_protein = PandasPdb().read_pdb(pdb_file)._df  # 读取文件
    protein = df_protein['ATOM']
    hetatm = df_protein['HETATM']
    data_df = pd.concat([protein, hetatm], axis=0)

    xyz, atom_index, grouped_xyz, fasta_index, align_list, chains, label = make_train_data(data_df, device, 'common')
    print(1)