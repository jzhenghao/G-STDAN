import torch as t
import torch
import math
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat  # einops 是一个用于操作PyTorch张量维度的库，来重新排列和变换张量的形状


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(BiLSTMEncoder, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

    def forward(self, x):
        # x: [seq_len, batch, input_dim]
        output, (hn, cn) = self.bilstm(x)
        # output: [seq_len, batch, hidden_dim*2]
        return output, (hn, cn)


def build_adjacency_matrix(positions, velocities, sigma=1.0):

    diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [batch, num_agents, num_agents, 2]
    dist = torch.norm(diff, dim=-1)  # [batch, num_agents, num_agents]
    dist_weight = torch.exp(-dist ** 2 / (2 * sigma ** 2))
    v1 = velocities.unsqueeze(2)  # [batch, num_agents, 1, 2]
    v2 = velocities.unsqueeze(1)  # [batch, 1, num_agents, 2]
    dot = (v1 * v2).sum(-1)
    norm1 = torch.norm(v1, dim=-1)
    norm2 = torch.norm(v2, dim=-1)
    cos_angle = dot / (norm1 * norm2 + 1e-6)
    angle_weight = (cos_angle + 1) / 2  # 归一化到[0,1]
    adj = dist_weight * angle_weight
    return adj


class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # x: [batch, num_agents, in_dim]
        # adj: [batch, num_agents, num_agents]
        agg = torch.bmm(adj, x)  # [batch, num_agents, in_dim]
        out = self.linear(agg)
        return F.relu(out)


class GDEncoder(nn.Module):  # 一个用于处理轨迹数据的编码器，为了轨迹预测或分析任务
    def __init__(self, args):
        super(GDEncoder, self).__init__()
        self.device = args['device']
        self.lstm_encoder_size = args['lstm_encoder_size']  # LSTM层的隐藏单元数
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.f_length = args['f_length']
        self.relu_param = args['relu']
        self.traj_linear_hidden = args['traj_linear_hidden']  # 轨迹数据通过第一个线性层时的隐藏单元数量
        self.use_maneuvers = args['use_maneuvers']
        self.use_elu = args['use_elu']
        self.use_spatial = args['use_spatial']  # 空间信息
        self.dropout = args['dropout']

        # traj encoder  线性层将输入特征的长度self.f_length映射到一个隐藏层的大小self.traj_linear_hidden
        self.linear1 = nn.Linear(self.f_length, self.traj_linear_hidden)
        self.lstm = nn.LSTM(self.traj_linear_hidden, self.lstm_encoder_size)

        # activation function
        if self.use_elu:
            self.activation = nn.ELU()  # ELU 是一种非线性激活函数，它可以加速收敛
        else:
            self.activation = nn.LeakyReLU(self.relu_param)  # LeakyReLU 激活函数，其斜率由 self.relu_param 指定

        self.qff = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.kff = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.vff = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.first_glu = GLU(input_size=self.n_head * self.att_out, hidden_layer_size=self.lstm_encoder_size,
                             dropout_rate=self.dropout)
        self.second_glu = GLU(input_size=self.n_head * self.att_out, hidden_layer_size=self.lstm_encoder_size,
                              dropout_rate=self.dropout)
        self.qt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.kt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.vt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.addAndNorm = AddAndNorm(self.lstm_encoder_size)
        # self.fc = nn.Linear(self.lstm_encoder_size * 2, self.lstm_encoder_size) # 这里的fc可能需要根据融合后的维度调整

        # BiLSTM 部分：输入维度改为 self.traj_linear_hidden
        self.bilstm = BiLSTMEncoder(input_dim=self.traj_linear_hidden,
                                    hidden_dim=64)  # BiLSTM 输出 128 (bidirectional * 64)

        # GCN 部分：SimpleGCNLayer 输入维度是 4 (位置+速度)，输出 64
        self.gcn_layer = SimpleGCNLayer(in_dim=4, out_dim=64)  # 位置(2)+速度(2) -> 64


        # 为了让 SpatialGatedFusion 处理，需要先对 GCN 输出进行一次线性变换使其与 BiLSTM 输出维度一致
        self.gcn_to_fusion = nn.Linear(64, 128)  # GCN 输出 64 -> 128

        self.attention_to_fusion = nn.Linear(self.lstm_encoder_size, 128)
        self.spatial_fusion1 = SpatialGatedFusion(feature_dim=128)
        self.spatial_fusion2 = SpatialGatedFusion(feature_dim=128)

        # 最终融合层：将第二层融合输出映射到 lstm_encoder_size
        self.final_fusion_fc = nn.Linear(128, self.lstm_encoder_size)

    # 它是模型的前向传播函数，用于执行输入数据通过网络层的计算过程。这个方法使用了多头注意力机制和LSTM网络来处理序列数据
    def forward(self, hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls):
        # 原始特征拼接 (保持不变)
        if self.f_length == 5:
            hist_orig = t.cat((hist, cls, va), -1)
            nbrs_orig = t.cat((nbrs, nbrscls, nbrsva), -1)
        elif self.f_length == 6:
            hist_orig = t.cat((hist, cls, va, lane), -1)
            nbrs_orig = t.cat((nbrs, nbrscls, nbrsva, nbrslane), -1)
        else:
            hist_orig = hist  # 如果 f_length 不是5或6，假设 hist 本身是完整的特征
            nbrs_orig = nbrs

        # self agent 特征通过 linear1 和 activation
        # hist_enc shape: [seq_len, batch, traj_linear_hidden]
        hist_enc = self.activation(self.linear1(hist_orig))


        # 注意力分支的输入是 hist_enc
        hist_hidden_enc, (_, _) = self.lstm(hist_enc)  # [seq_len, batch, lstm_encoder_size]
        # 注意力分支后续计算保持不变，直到 attention_output
        hist_hidden_enc_permuted = hist_hidden_enc.permute(1, 0, 2)  # [batch, seq_len, lstm_encoder_size]

        nbrs_enc = self.activation(self.linear1(nbrs_orig))
        nbrs_hidden_enc, (_, _) = self.lstm(nbrs_enc)
        mask = mask.view(mask.size(0), mask.size(1) * mask.size(2), mask.size(3))
        mask = repeat(mask, 'b g s -> t b g s', t=self.in_length)
        soc_enc = t.zeros_like(mask).float()
        soc_enc = soc_enc.masked_scatter_(mask, nbrs_hidden_enc)

        # Spatial Attention (从 Attention 分支分离出来以清晰表示) - 输入使用 hist_hidden_enc_permuted 和 soc_enc
        query_spa = self.qff(hist_hidden_enc_permuted)  # [batch, seq_len, n_head * att_out]
        _, _, embed_size_spa = query_spa.shape
        query_spa = t.cat(t.split(t.unsqueeze(query_spa, 2), int(embed_size_spa / self.n_head), -1), 1)
        keys_spa = t.cat(t.split(self.kff(soc_enc), int(embed_size_spa / self.n_head), -1), 0).permute(1, 0, 3, 2)
        values_spa = t.cat(t.split(self.vff(soc_enc), int(embed_size_spa / self.n_head), -1), 0).permute(1, 0, 2, 3)
        a_spa = t.matmul(query_spa, keys_spa)  # [batch, n_head, seq_len, num_nbrs]
        a_spa /= math.sqrt(self.lstm_encoder_size)
        a_spa = t.softmax(a_spa, -1)
        values_attn_spa = t.matmul(a_spa, values_spa)  # [batch, n_head, seq_len, att_out]
        values_attn_spa = t.cat(t.split(values_attn_spa, int(hist_orig.shape[0]), 1),
                                -1)  # [batch, seq_len, n_head*att_out]

        spa_values, _ = self.first_glu(values_attn_spa)  # [batch, seq_len, lstm_encoder_size]

        if spa_values.dim() == 4 and spa_values.shape[2] == 1:
            spa_values = spa_values.squeeze(2)

        attention_intermediate = self.addAndNorm(hist_hidden_enc_permuted,
                                                 spa_values)  # [batch, seq_len, lstm_encoder_size]

        # Temporal Attention - 输入使用 attention_intermediate
        qt = t.cat(t.split(self.qt(attention_intermediate), int(embed_size_spa / self.n_head), -1),
                   0)  # [n_head*batch, seq_len, att_out]
        kt = t.cat(t.split(self.kt(attention_intermediate), int(embed_size_spa / self.n_head), -1), 0).permute(0, 2,
                                                                                                               1)  # [n_head*batch, att_out, seq_len]
        vt = t.cat(t.split(self.vt(attention_intermediate), int(embed_size_spa / self.n_head), -1),
                   0)  # [n_head*batch, seq_len, att_out]

        a_temp = t.matmul(qt, kt)  # [n_head*batch, seq_len, seq_len]
        a_temp /= math.sqrt(self.lstm_encoder_size)
        a_temp = t.softmax(a_temp, -1)
        values_attn_temp = t.matmul(a_temp, vt)  # [n_head*batch, seq_len, att_out]
        values_attn_temp = t.cat(t.split(values_attn_temp, int(hist_orig.shape[1]), 0),
                                 -1)  # [batch, seq_len, n_head*att_out]

        time_values, _ = self.second_glu(values_attn_temp)  # [batch, seq_len, lstm_encoder_size]
        attention_output = attention_intermediate  # 暂定 Spatial Attention residual output 为 Attention 分支输出

        bilstm_out, _ = self.bilstm(hist_enc)  # [seq_len, batch, 128]
        bilstm_last = bilstm_out[-1]  # 取最后一个时间步的输出 [batch, 128]
        positions = hist[-1][..., :2]  # [batch, num_agents, 2]
        velocities = va[-1][..., :2]  # [batch, num_agents, 2]

        # 自动扩展维度，保证至少3维
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        if velocities.dim() == 2:
            velocities = velocities.unsqueeze(0)

        # 自动裁剪到相同agent数
        min_agents = min(positions.shape[1], velocities.shape[1])
        positions = positions[:, :min_agents, :]
        velocities = velocities[:, :min_agents, :]
        gcn_spatial_input = torch.cat([positions, velocities], dim=-1)  # [batch, min_agents, 4]
        adj = build_adjacency_matrix(positions, velocities)  # [batch, min_agents, min_agents]
        gcn_out = self.gcn_layer(gcn_spatial_input, adj)  # [batch, min_agents, 64]

        # GCN 输出均值池化，然后通过线性层使其维度与 BiLSTM 输出一致 (128)
        gcn_feat = gcn_out.mean(dim=1)  # [batch, 64]
        gcn_feat_aligned = self.gcn_to_fusion(gcn_feat)  # [batch, 128]

        # ---------------- 空间门控融合 (两层) ----------------
        # 第一层融合输入：BiLSTM 输出 (128), GCN 对齐后的输出 (128)

        # 添加批次维度对齐检查和处理
        if bilstm_last.shape[0] != gcn_feat_aligned.shape[0]:
            # print(f"[DEBUG] Batch dimension mismatch before spatial_fusion1: bilstm_last batch {bilstm_last.shape[0]} vs gcn_feat_aligned batch {gcn_feat_aligned.shape[0]}")
            if gcn_feat_aligned.shape[0] == 1:
                # 如果 gcn_feat_aligned 的批次维度是 1，将其扩展到 bilstm_last 的批次维度
                gcn_feat_aligned = gcn_feat_aligned.expand(bilstm_last.shape[0], -1)
                # print(f"[DEBUG] gcn_feat_aligned shape after expanding batch: {gcn_feat_aligned.shape}")
            else:
                # 如果批次维度不匹配且 gcn_feat_aligned 的批次不是 1，则可能是其他问题
                raise ValueError(
                    f"Batch dimension mismatch before spatial_fusion1: bilstm_last batch {bilstm_last.shape[0]} vs gcn_feat_aligned batch {gcn_feat_aligned.shape[0]}")

        # print(f"[DEBUG] Before spatial_fusion1: bilstm_last shape: {bilstm_last.shape}, gcn_feat_aligned shape: {gcn_feat_aligned.shape}")
        fusion1_output = self.spatial_fusion1(bilstm_last, gcn_feat_aligned)  # [batch, 128]

        # 第二层融合输入：第一层融合输出 (128), Attention 分支对齐后的输出 (128)
        # attention_output shape: [batch, seq_len, lstm_encoder_size]
        # 我们需要 attention 分支的最后一个时间步的特征，并对齐到 128
        attention_feat_for_fusion2 = attention_output[:, -1, :]  # [batch, lstm_encoder_size]
        attention_feat_aligned_for_fusion2 = self.attention_to_fusion(attention_feat_for_fusion2)  # [batch, 128]

        fusion2_output = self.spatial_fusion2(fusion1_output, attention_feat_aligned_for_fusion2)  # [batch, 128]

        # 最终输出：将第二层融合输出映射到 lstm_encoder_size
        # 这里的 lstm_encoder_size 是 Generator 期望的输入维度
        values = self.final_fusion_fc(fusion2_output)  # [batch, lstm_encoder_size]
        # 返回 GDEncoder 的输出
        return values


def outputActivation(x):  # 处理二维高斯分布的参数
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = t.exp(sigX)
    sigY = t.exp(sigY)
    rho = t.tanh(rho)
    out = t.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out  # 包含所有参数的输出张量


class AddAndNorm(nn.Module):
    def __init__(self, hidden_layer_size):  # 对张量进行层归一化
        super(AddAndNorm, self).__init__()

        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1, x2, x3=None):  # 前向传播函数，定义了如何计算输出
        if x3 is not None:
            x = t.add(t.add(x1, x2), x3)
        else:
            x = t.add(x1, x2)
        return self.normalize(x)


class Decoder(nn.Module):  # 处理序列数据解码
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.relu_param = args['relu']
        self.use_elu = args['use_elu']
        self.use_maneuvers = args['use_maneuvers']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.encoder_size = args['lstm_encoder_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.device = args['device']
        self.cat_pred = args['cat_pred']
        self.use_mse = args['use_mse']
        self.lon_length = args['lon_length']
        self.lat_length = args['lat_length']
        if self.use_maneuvers or self.cat_pred:  # 使用驾驶策略（self.use_maneuvers）和预测结果拼接（self.cat_pred）
            self.mu_f = 16
        else:
            self.mu_f = 0
        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)

        self.lstm = t.nn.LSTM(self.encoder_size, self.encoder_size)  # 初始化 LSTM 层
        if self.use_mse:
            self.linear1 = nn.Linear(self.encoder_size, 2)
        else:
            self.linear1 = nn.Linear(self.encoder_size, 5)
        self.lat_linear = nn.Linear(self.lat_length, 8)  # 初始化横向和纵向线性层
        self.lon_linear = nn.Linear(self.lon_length, 8)

        self.dec_linear = nn.Linear(self.encoder_size + self.lat_length + self.lon_length, self.encoder_size)

    def forward(self, dec, lat_enc, lon_enc):  # 经过编码器处理的序列数据，作为解码器的初始隐藏状态或上下文

        if self.use_maneuvers or self.cat_pred:  # 将 lat_enc 和 lon_enc 进行扩展和拼接，以便在解码过程中使用这些特征
            lat_enc = lat_enc.unsqueeze(1).repeat(1, self.out_length, 1).permute(1, 0, 2)
            lon_enc = lon_enc.unsqueeze(1).repeat(1, self.out_length, 1).permute(1, 0, 2)
            dec = t.cat((dec, lat_enc, lon_enc), -1)  # t.cat 将 dec、lat_enc 和 lon_enc 在最后一个维度上进行拼接
            dec = self.dec_linear(dec)  # 整合来自编码器、横向和纵向的特征
        h_dec, _ = self.lstm(dec)  # 解码器的隐藏状态
        fut_pred = self.linear1(h_dec)  # 从 LSTM 层的输出 h_dec 生成的未来预测
        if self.use_mse:
            return fut_pred
        else:
            return outputActivation(fut_pred)


class Generator(nn.Module):  # 生成和解码车辆轨迹的神经网络结构
    def __init__(self, args):
        super(Generator, self).__init__()
        self.device = args['device']
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.f_length = args['f_length']
        self.relu_param = args['relu']
        self.train_flag = args['train_flag']
        self.traj_linear_hidden = args['traj_linear_hidden']
        self.use_maneuvers = args['use_maneuvers']
        self.lat_length = args['lat_length']
        self.lon_length = args['lon_length']
        self.use_elu = args['use_elu']
        self.use_true_man = args['use_true_man']
        self.Decoder = Decoder(args=args)  # 嵌套了一个解码器
        self.mu_fc1 = t.nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)  # 线性变换层
        self.mu_fc = t.nn.Linear(self.n_head * self.att_out, self.lstm_encoder_size)  # 重新映射回原始的 LSTM 编码器特征维度
        self.op_lat = t.nn.Linear(self.lstm_encoder_size, self.lat_length)  # 将 LSTM 编码器的输出特征转换为横向（lateral）特征表示
        self.op_lon = t.nn.Linear(self.lstm_encoder_size, self.lon_length)  # 将特征转换为纵向（longitudinal）特征表示

        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)
        self.normalize = nn.LayerNorm(self.lstm_encoder_size)  # 归一化层
        # 创建了一个三维张量
        self.mapping = t.nn.Parameter(t.Tensor(self.in_length, self.out_length, self.lat_length + self.lon_length))
        nn.init.xavier_uniform_(self.mapping, gain=1.414)  # Xavier 均匀初始化方法
        self.manmapping = t.nn.Parameter(t.Tensor(self.in_length, 1))  # 创建一个二维张量，并将其作为参数
        nn.init.xavier_uniform_(self.manmapping, gain=1.414)  #

    def forward(self, values, lat_enc, lon_enc):  # Generating predicted results
        # values from GDEncoder: [batch, feature] e.g. [32, 64]

        # Derive maneuver_state for maneuver prediction
        # Assumes values is [batch, feature] from fusion layer
        maneuver_state = values  # [batch, feature]

        maneuver_state = self.activation(self.mu_fc1(maneuver_state))
        maneuver_state = self.activation(self.normalize(self.mu_fc(maneuver_state)))
        lat_pred = F.softmax(self.op_lat(maneuver_state), dim=-1)
        lon_pred = F.softmax(self.op_lon(maneuver_state), dim=-1)

        if self.train_flag:
            if self.use_true_man:
                lat_man = t.argmax(lat_enc, dim=-1).detach()  # argmax 找到每个样本的最高概率策略索引 detach() 用于将计算图从当前的 lat_man 分离
                lon_man = t.argmax(lon_enc, dim=-1).detach()
            else:
                lat_man = t.argmax(lat_pred, dim=-1).detach().unsqueeze(1)  # 模型预测的横向概率分布 lat_pred 中找到最可能的策略索引，并增加一个维度
                lon_man = t.argmax(lon_pred, dim=-1).detach().unsqueeze(1)
                lat_enc_tmp = t.zeros_like(lat_pred)  # 创建与预测概率分布形状相同的零张量，用于存放策略索引
                lon_enc_tmp = t.zeros_like(lon_pred)
                lat_man = lat_enc_tmp.scatter_(1, lat_man, 1)  # 使用 scatter_ 在零张量中放置策略索引，1 表示在第二个维度（索引的维度）上放置值
                lon_man = lon_enc_tmp.scatter_(1, lon_man, 1)

            # 打印调试信息
            # print(f"[DEBUG] self.mapping shape: {self.mapping.shape}")
            # print(f"[DEBUG] lat_man shape: {lat_man.shape}, lon_man shape: {lon_man.shape}")

            index = t.cat((lat_man, lon_man), dim=-1).permute(-1, 0)  # 横向和纵向的策略索引拼接起来，并重新排列维度
            # print(f"[DEBUG] index shape after permute: {index.shape}")

            # 计算 mapping
            mapping_intermediate = t.matmul(self.mapping, index)
            # print(f"[DEBUG] mapping_intermediate shape: {mapping_intermediate.shape}")

            mapping = F.softmax(mapping_intermediate.permute(2, 1, 0), dim=-1)
            # print(f"[DEBUG] mapping shape after permute and softmax: {mapping.shape}")

            # 确保 values 的维度正确
            if values.dim() == 2:  # values 是 [batch, feature]
                values = values.unsqueeze(1).repeat(1, self.in_length, 1)  # 扩展为 [batch, in_length, feature]
                # print(f"[DEBUG] values shape after unsqueeze and repeat: {values.shape}")

            # 确保 mapping 和 values 的批次维度匹配
            if mapping.shape[0] != values.shape[0]:
                if mapping.shape[0] == 1:
                    mapping = mapping.expand(values.shape[0], -1, -1)
                    # print(f"[DEBUG] mapping shape after expanding batch dimension: {mapping.shape}")
                else:
                    raise ValueError(f"批次维度不匹配: mapping {mapping.shape[0]} vs values {values.shape[0]}")

            # 执行矩阵乘法
            # print(f"[DEBUG] Before matmul - mapping shape: {mapping.shape}, values shape: {values.shape}")
            dec = t.matmul(mapping, values).permute(1, 0, 2)  # [out_length, batch, feature]
            # print(f"[DEBUG] After matmul and permute - dec shape: {dec.shape}")

            if self.use_maneuvers:
                fut_pred = self.Decoder(dec, lat_enc, lon_enc)  # 调用 self.Decoder 这个解码器模块
                return fut_pred, lat_pred, lon_pred  # 返回未来的轨迹预测以及横向和纵向的概率分布 允许在模型输出中同时获得预测轨迹和对应的策略概率
            else:
                fut_pred = self.Decoder(dec, lat_pred, lon_pred)  # 使用模型预测的横向和纵向策略概率分布 lat_pred 和 lon_pred 作为输入
                return fut_pred, lat_pred, lon_pred
        else:
            out = []  # 空列表 out用于存储生成的多模态轨迹预测结果
            for k in range(self.lon_length):  # 嵌套循环遍历横向 (lat) 和纵向 (lon) 策略的所有可能索引
                for l in range(self.lat_length):
                    lat_enc_tmp = t.zeros_like(lat_enc)  # 创建与 lat_enc 和 lon_enc 形状相同的零张量 lat_enc_tmp 和 lon_enc_tmp
                    lon_enc_tmp = t.zeros_like(lon_enc)
                    lat_enc_tmp[:, l] = 1  # 将对应索引 l 和 k 的位置设置为 1
                    lon_enc_tmp[:, k] = 1
                    index = t.cat((lat_enc_tmp, lon_enc_tmp), dim=-1).permute(-1, 0)  # 编码张量沿着最后一个维度拼接重新排列维度以适应后续的矩阵乘法
                    mapping = F.softmax(t.matmul(self.mapping, index).permute(2, 1, 0), dim=-1)  # 映射概率分布

                    # 确保 values 的维度正确
                    if values.dim() == 2:
                        values = values.unsqueeze(1).repeat(1, self.in_length, 1)

                    # 确保 mapping 和 values 的批次维度匹配
                    if mapping.shape[0] != values.shape[0]:
                        if mapping.shape[0] == 1:
                            mapping = mapping.expand(values.shape[0], -1, -1)

                    dec = t.matmul(mapping, values).permute(1, 0, 2)  # 将映射概率分布与 values 进行矩阵乘法，并重新排列维度，得到解码器的输入 dec
                    fut_pred = self.Decoder(dec, lat_enc_tmp, lon_enc_tmp)  # 生成轨迹预测
                    out.append(fut_pred)  # 将生成的轨迹预测 fut_pred 添加到列表 out
            return out, lat_pred, lon_pred


class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = {
            'input_dim': config.get('gcn_input_dim', 256),
            'hidden_dim': config.get('gcn_hidden_dim', 256),
            'output_dim': config.get('gcn_output_dim', 256),
            'n_heads': config.get('gcn_heads', 4),
            'topk_threshold': config.get('gcn_topk', 0.5),
            'activation': config.get('gcn_act', 'leaky_relu')
        }
        self.adjacency_learner = nn.Identity()
        self.edge_encoder = nn.Sequential(
            nn.Linear(2 * self.config['input_dim'], 1),
            nn.Identity()  # 保持维度兼容性
        )
        self.gcn_cells = nn.ModuleList([
            PlaceholderGCNLayer(
                in_dim=self.config['input_dim'],
                out_dim=self.config['hidden_dim'],
                activation=self.config['activation']
            ) for _ in range(2)
        ])
        self.temporal_sync = nn.Identity()
        self.spatial_norm = nn.LayerNorm(self.config['output_dim'])

    def _build_dynamic_graph(self, node_states: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            node_states.size(0),
            node_states.size(1),
            node_states.size(2),
            node_states.size(2),
            device=node_states.device
        )


class PlaceholderGCNLayer(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 activation: str = 'leaky_relu'):
        super().__init__()
        self.feature_proj = nn.Identity()
        self.attention_mech = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=1,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=in_dim,
            vdim=in_dim
        )
        self.act = nn.Identity()

    def forward(self,
                ego: torch.Tensor,
                neighbors: torch.Tensor) -> [torch.Tensor, torch.Tensor]:

        attn_out, _ = self.attention_mech(
            ego.unsqueeze(1),
            neighbors.permute(1, 0, 2, 3),
            neighbors.permute(1, 0, 2, 3)
        )
        return self.act(attn_out.squeeze(1)), neighbors


# gate
class GLU(nn.Module):
    # Gated Linear Unit+
    def __init__(self,
                 input_size,
                 hidden_layer_size,
                 dropout_rate=None,
                 ):
        super(GLU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_layer = t.nn.Linear(input_size, hidden_layer_size)
        self.gated_layer = t.nn.Linear(input_size, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # 前向传播
        if self.dropout_rate is not None:
            x = self.dropout(x)  # 使用dropout层来正则化输入x，防止过拟合
        activation = self.activation_layer(x)  # 一个线性层处理输入x，生成激活信号
        gated = self.sigmoid(self.gated_layer(x))  # 通过另一个线性层self.gated_layer处理输入x，生成门控信号
        return t.mul(activation, gated), gated


class SpatialGatedFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # 这里的输入维度是所有分支特征的拼接维度
        # 目前只有 BiLSTM 和 GCN，所以是 feature_dim * 2
        self.gate_layer = nn.Linear(feature_dim * 2, feature_dim * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, bilstm_feat, gcn_feat):
        # bilstm_feat: [batch, feature_dim]
        # gcn_feat: [batch, feature_dim]

        # 拼接特征
        combined_feat = torch.cat([bilstm_feat, gcn_feat], dim=-1)  # [batch, feature_dim * 2]

        # 计算门控权重
        gates = self.sigmoid(self.gate_layer(combined_feat))  # [batch, feature_dim * 2]

        # 分割门控权重
        gate_bilstm, gate_gcn = torch.split(gates, bilstm_feat.shape[-1], dim=-1)  # 各自 [batch, feature_dim]

        # 门控融合
        fused_feat = gate_bilstm * bilstm_feat + gate_gcn * gcn_feat

        return fused_feat  # [batch, feature_dim]