"""
64组4头-最多5000行-80g显存
32组4头-最多8000行
32组2头-最多11000行
16组2头-12000未见显存不足
"""

batch_size = 1  # 一次只能训练一个文件
epochs = 100  # 训练的轮次
max_len = 8000  # 文件最大长度

learning_rate = 0.001  # 学习率
attnLayersNum = 8  # 编码器的数量
heads = 8  # 注意力头数量
drops = 0.1  # 丢弃

list_range = [[3, 4, 5], [8, 16, 24]]
#list_range = [[4, 5, 6], [16, 24, 32]]  # 456A范围内分别取16、32、64个点
# list_range = [[4], [16]]
em_dim = 64  # 初始维度
fasta_dim = 16  # Fasta维度
localdims = len(list_range[0]) * em_dim  # 这个公式不要改

aotm_em = 32
chain_em = 8
loss_weight = 0.65  # 损失函数的加权值
fgama = 2
confidence_level = 0.5  # 输出结果的置信度
train_label_pockets_radius = 4.5  # 4.5A范围内的原子作为口袋，比例高一点
is_continues_cof = False
# continue_models = './model4_all_continue_h_new4/model_file/model_parameter_0.pkl'
continue_models = './self-model-celoss25-lr0001-continue4/model_file/model_parameter_5.pkl'


if __name__ == '__main__':
    import torch
    a = torch.ones(10, 2)
    b = torch.ones(10, 2)
    c = torch.ones(10, 2)

    d = torch.cat([a, b, c], dim=1)
    print(d)
