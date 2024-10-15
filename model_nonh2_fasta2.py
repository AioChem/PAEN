import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch


class TNetin(nn.Module):
    """
    coord-trans
    3D坐标信息变换模块
    """
    def __init__(self, k):
        super(TNetin, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        in_size: Batch,Point,Channel (1 n 3)
        return: 1,3,3
        """
        batchsize = x.size()[0]
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.fc1(x))
        x = self.fc3(self.relu(self.fc2(x)))

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class An(nn.Module):
    """
    ADD Norm
    """
    def __init__(self, anem_dim):
        super(An, self).__init__()
        # self.drop = nn.Dropout(p=drops)
        self.ly1 = nn.LayerNorm(anem_dim)

    def forward(self, x1, x2):
        return self.ly1(x1 + x2)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=1, stride=1, bias=True),
            # nn.BatchNorm1d(out_channels),
            # nn.LeakyReLU(negative_slope=1, inplace=True)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 1 32 n
        x = self.double_conv(x)
        return x.permute(0, 2, 1)


class Fw(nn.Module):
    """
    Feed froward
    input:
    """

    def __init__(self, fin_dim, fout_dim, fmid_dim=None):
        super(Fw, self).__init__()
        if not fmid_dim:
            fmid_dim = fout_dim
        self.LRL = nn.Sequential(
            nn.Linear(fin_dim, fmid_dim),
            nn.ReLU(),
            nn.Linear(fmid_dim, fout_dim),
        )
    def forward(self, x):
        # input_size B N C   (1, 10, 3)
        return self.LRL(x)


class EncodersAttn(nn.Module):
    def __init__(self, in_dims, eout_dims, eheaders, edrops):
        super(EncodersAttn, self).__init__()
        # N, L, E_q   batch , length , d_model
        self.attn = nn.MultiheadAttention(embed_dim=in_dims, num_heads=eheaders, dropout=edrops, batch_first=True)
        self.an1 = An(in_dims)
        self.fw1 = Fw(fin_dim=in_dims, fout_dim=in_dims, fmid_dim=in_dims*2)
        self.an2 = An(in_dims)
        self.fw2 = Fw(fin_dim=in_dims, fout_dim=eout_dims, fmid_dim=(eout_dims+in_dims)//2)

    def forward(self, x):
        # 自查询，获取全局信息
        attn_out, _ = self.attn(x, x, x)
        # 第一层的残差连接
        attn_ly_out1 = self.an1(x, attn_out)
        # Feed-Forward
        ln1_out = self.fw1(attn_ly_out1)
        # 第二层残差
        attn_ly_out2 = self.an2(attn_ly_out1, ln1_out)
        # Feed-Forward
        ln2_out = self.fw2(attn_ly_out2)
        return ln1_out, ln2_out


class Input_ly_atom(nn.Module):
    """
    返回原子特征
    t2_out：第一层的点的特征输出
    """

    def __init__(self, laem_dim):
        super(Input_ly_atom, self).__init__()
        self.t1_net = TNetin(k=3)
        self.em_atom = nn.Embedding(5, 3)  # 将原子name的编码扩充到3维度
        self.fw1 = Fw(fin_dim=3, fout_dim=laem_dim, fmid_dim=laem_dim*2)  # 3+16+5原子编码维度加xyz，总共9个维度

    def forward(self, xyz, atom_fasta):
        # 将xyz先对齐一次
        t1_out = torch.matmul(xyz, self.t1_net(xyz))  # 1 n 3
        # 编码原子类型
        atom_pos_em = self.em_atom(atom_fasta.long())  # input: 1, n  return:1 n 32
        # 合并原子名称和坐标
        toutem = torch.add(t1_out, atom_pos_em)
        # 经过fw层
        t1out_du = self.fw1(toutem)  # 1 n 64

        return t1out_du  # out_size:1, n ,c


class Input_ly_fasta(nn.Module):
    """
    返回序列
    """
    def __init__(self, ifasta_em_dim, iem_dims):
        super(Input_ly_fasta, self).__init__()
        self.em_atom = nn.Embedding(21, ifasta_em_dim)  # 将氨基酸名字的编码扩充到32维度
        self.fw = Fw(fin_dim=ifasta_em_dim+iem_dims, fout_dim=iem_dims, fmid_dim=iem_dims*2)
        # self.dv1 = DoubleConv(in_channels=fasta_em_dim+local_dims, out_channels=local_dims, mid_channels=local_dims*2)
        # self.an1 = An(em_dim=local_dims)
        # self.ln1 = nn.Linear(local_dims, em_dims)
        self.indexp = AlignCount()

    def forward(self, lasts_ous_xyz, centroids, fasta):
        # 编码氨基酸序列
        fasta_pos_em = self.em_atom(fasta.long())  # 1 group fe  转为多少维度定义在config里面，目前看起来32挺好
        # TODO CUDA
        # a = lasts_ous_xyz.device.type
        if lasts_ous_xyz.device.type != 'cpu':
            # lastXyz_t = torch.empty(0, requires_grad=True).cuda()
            # torch.empty_like(other_tensor)
            lastXyz_t = torch.zeros(fasta_pos_em.size()[1], lasts_ous_xyz.size()[2]).cuda()
        else:
            lastXyz_t = torch.zeros(fasta_pos_em.size()[1], lasts_ous_xyz.size()[2])
        # 将之前的1 n 64 原子信息对应进来
        for i in centroids:
            out_xyz_last = self.indexp.index_point(lasts_ous_xyz, i)  # 4A 16 个点  1 F 16 64
            out_xyz_last = out_xyz_last.squeeze(0)  # F 16 64
            out_xyz_last = torch.max(out_xyz_last, dim=1)[0]  # F 64
            lastXyz_t = torch.add(lastXyz_t, out_xyz_last)  # 1 F 64
            # lastXyz_t = torch.cat([lastXyz_t, out_xyz_last.unsqueeze(0)], dim=2)  # 1 F 64
        out_cat = torch.cat([fasta_pos_em, lastXyz_t.unsqueeze(0)], dim=2)  # 1 F 16 | 1 F 128
        # out = self.dv1(out_cat)  # 1 F C

        # out = self.an1(lastXyz_t, out)
        out = self.fw(out_cat)
        # d = out.detach().numpy()
        # print(1)
        return out


class AlignCount(nn.Module):
    def __init__(self):
        super(AlignCount, self).__init__()

    def aligns(self, features, count_list):
        data_cp = []
        # 做查询，扩充氨基酸的点信息，分配给每个点
        for index, i in enumerate(features):
            news = i.repeat(count_list[index]).view(count_list[index], -1)
            data_cp.extend(news)
        data_cp_end = torch.stack(data_cp)
        return data_cp_end.unsqueeze(0)

    def index_point(self, points, idx):
        # 根据中心点的index分配特征
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points


class AlignFormer(nn.Module):
    def __init__(self, aem_dim, aout_dim, aheads, adrop):
        super(AlignFormer, self).__init__()
        self.atomEnc = EncodersAttn(in_dims=aem_dim, eout_dims=aout_dim, eheaders=aheads, edrops=adrop)
        # self.ans = An(aem_dim)

        self.fasta_cross_attn = nn.MultiheadAttention(embed_dim=aem_dim, num_heads=aheads, dropout=adrop, batch_first=True)
        self.ans2 = An(aem_dim)
        self.lrl = Fw(fin_dim=aem_dim, fout_dim=aout_dim, fmid_dim=(aout_dim+aem_dim)//2)
        # self.fastaEnc = EncodersAttn(in_dims=aem_dim, eout_dims=aout_dim, eheaders=aheads, edrops=adrop)
        self.aligns1 = AlignCount()

    def forward(self, out_atom, out_fasta, query_list, centroids):
        # 先扩充fasta_out
        align_fasta = self.aligns1.aligns(out_fasta.squeeze(0), query_list)
        # cat结果
        # atom_in = torch.cat([out_atom, align_fasta], dim=2)  # 1 n 64*3
        atom_in = torch.add(out_atom, align_fasta)
        # Transformer，查询全局信息后降维
        atom_trans, atom_out = self.atomEnc(atom_in)  # 1 n 64
        # print(1)
        # TODO CUDA
        if out_atom.device.type != 'cpu':
            out_align_atom = torch.zeros(out_fasta.size()[1], atom_trans.size()[2]).cuda()
        else:
            out_align_atom = torch.zeros(out_fasta.size()[1], atom_trans.size()[2])
        # range-search
        for i in centroids:
            out_xyz_last = self.aligns1.index_point(atom_trans, i)  # 4A 16 个点  1 F 16 64
            out_xyz_last = out_xyz_last.squeeze(0)  # F 16 64
            # out_xyz_last = dv(out_xyz_last)  # F 1 64
            out_xyz_last = torch.max(out_xyz_last, dim=1)[0]  # F 64
            # out_xyz_last = out_xyz_last.unsqueeze(0)
            out_align_atom = torch.add(out_align_atom, out_xyz_last)  # F 1 64+x
        # q
        fasta_q = torch.add(out_fasta, out_align_atom)
        fasta_trans, _ = self.fasta_cross_attn(fasta_q, fasta_q, fasta_q)
        fasta_trans = self.ans2(fasta_q, fasta_trans)
        fasta_trans = self.lrl(fasta_trans)
        # cat
        # fasta_cat = torch.add(out_align_tink, out_fasta)
        # 再Transformer
        # fasta_trans = self.fastaEnc(fasta_cat)  # 1 F 128
        return atom_out, fasta_trans  # 1 n 128   and 1 F 128


class Decoders(nn.Module):
    def __init__(self, indim):
        super(Decoders, self).__init__()
        self.dv1 = nn.Sequential(
            nn.Conv1d(indim, indim//2, kernel_size=3, padding=1, stride=1,bias=True),
            nn.BatchNorm1d(indim//2),
            # nn.LeakyReLU(negative_slope=1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(indim//2, 1, kernel_size=3, padding=1, stride=1, bias=True),
        )
        self.aligns2 = AlignCount()

    def forward(self, out_atom, out_fasta, count_list):
        # 扩充fasta
        out_fasta = self.aligns2.aligns(out_fasta.squeeze(0), count_list)
        # cat
        # out_cat = torch.cat([out_atom, out_fasta], dim=2)
        out_cat = torch.add(out_atom, out_fasta)
        # conv
        out_results = self.dv1(out_cat.permute(0, 2, 1))
        # TODO look
        #outdd = torch.sigmoid(out_results)
        #d1 = outdd.detach().numpy().flatten()
        #d2 = pd.DataFrame(d1)
        return out_results.squeeze(0).transpose(1, 0)  # n, 1


class PbstNet(nn.Module):
    def __init__(self, em_dim, fasta_em_dim, heads, drops, attnLayerNums):
        super(PbstNet, self).__init__()
        self.inpusLy_atom = Input_ly_atom(em_dim)
        self.inpusLy_fasta = Input_ly_fasta(fasta_em_dim, em_dim)
        self.AlignFormer_list = nn.ModuleList()
        self.sta = 1
        for index in range(attnLayerNums):
            """
            attnLayerNums 必须是偶数层
            """
            if index < attnLayerNums / 2:
                self.AlignFormer_list.append(AlignFormer(em_dim * self.sta, em_dim * self.sta * 2, heads, drops))
                self.sta *= 2
            else:
                self.AlignFormer_list.append(AlignFormer(em_dim * self.sta, em_dim * self.sta // 2, heads, drops))
                self.sta //= 2

        self.dec = Decoders(em_dim)

    def forward(self, xyz, atom_index, grouped_xyz_index, fasta_index, my_dict):
        # 返回点的特征
        out_atom = self.inpusLy_atom(xyz, atom_index)  # 1 n 64
        # 返回序列特征
        out_fasta = self.inpusLy_fasta(out_atom, grouped_xyz_index, fasta_index)  # 1 F X
        # fasta Former
        for encs in self.AlignFormer_list:
            out_atom, out_fasta = encs(out_atom, out_fasta, my_dict, grouped_xyz_index)

        outs = self.dec(out_atom, out_fasta, my_dict)

        # outsig = torch.sigmoid(outs)
        # outsd = outsig.cpu().detach().numpy()
        return outs


if __name__ == '__main__':
    import pandas as pd
    from makeData import make_train_data, opti_pdbs
    from utils import outcome_valid_sigmoid, count_pockets_label_no_water
    from utils import outcome_valid_sigmoid
    #from configs import *
    device = 'cpu'
    pdb_file = 'F:\\PBST-net\\pdb_datas\\tests_sets\\1Z95\\1Z95.pdb'
    ligand_df = pd.read_csv('./pdb_datas/ligand_het.csv')
    # 处理数据
    protein_data, hetatm_data = opti_pdbs(pdb_file, ligand_df)
    # 计算标签
    label = count_pockets_label_no_water(protein_data, hetatm_data, 4.5, device, 'common')  # (F, 1)
    # 实例化模型
    models = PbstNet(em_dim=64, fasta_em_dim=16, heads=4, drops=0.1, attnLayerNums=6).to(device)

    #load_model = torch.load('./modelsPTH/self-model-celoss25-lr0001-continue4/model_file/model_parameter_5.pkl')
    #models.load_state_dict(load_model['model_state_dict'])
    # 拿到模型输入信息
    xyz, atom_index, grouped_xyz, fasta_index, align_list = make_train_data(protein_data, device)

    # 开始预测
    predict = models(xyz, atom_index, grouped_xyz, fasta_index, align_list)
    # 计算指标
    miou, precision, recall, f1, acc, mcc, auroc = outcome_valid_sigmoid(predict, label, 0.5)
    print(miou)
    print(precision)
    print(recall)
    print(acc)
