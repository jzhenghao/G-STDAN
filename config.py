args = {}
import random
import numpy as np
import torch as t
import model5f_mult as model
args['path'] = 'checkpoint/true_1/'
args['data_dir'] = 'data/'  # 数据目录路径
# -------------------------------------------------------------------------
# 参数设置
seed = 72
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

args['learning_rate'] = 0.0005
dataset = "ngsim"  # highd ngsim
args['Angle threshold'] = 15
args['distance threshold'] = 100

args['num_worker'] = 0
args['device'] = device
args['lstm_encoder_size'] = 64  # LSTM的隐藏层数
args['lstm_decoder_size'] = 64  # 解码器LSTM的隐藏层数
args['n_head'] = 4
args['att_out'] = 48  # 注意力机制输出的维度
args['in_length'] = 16  # 输入序列的维度
args['out_length'] = 25
# 在config.py中添加以下参数
args.update({
    'mix_decay_epochs': 20,
    'min_mse_weight': 0.1,
})

args['f_length'] = 5
args['traj_linear_hidden'] = 32
args['batch_size'] = 32
args['use_elu'] = True
args['dropout'] = 0.2
args['relu'] = 0.1
args['lat_length'] = 3
args['lon_length'] = 3  # 2
args['use_true_man'] = False
args['epoch'] = 30
args['use_spatial'] = False


# 多模态
args['use_maneuvers'] = False
# 单模态是否拼接预测意图
args['cat_pred'] = True
args['use_mse'] = False
args['pre_epoch'] = 6
args['val_use_mse'] = True