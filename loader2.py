from __future__ import print_function, division  # print_function 确保使用Python 3的打印语法 division 使得除法运算 / 总是返回浮点数结果
from torch.utils.data import Dataset  # Dataset 类是PyTorch中用于定义自定义数据集的基类
import scipy.io as scp  # scipy.io 模块提供了用于读写MATLAB文件
import numpy as np  # NumPy是一个广泛使用的科学计算库
import torch
import h5py  # HDF5（一种用于存储和组织大量数据的文件格式）
from config import args  # config 模块定义了一个包含程序配置参数的 args 对象
import time


class NgsimDataset(Dataset):  # 创建一个数据集

    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3)):
        self.D = scp.loadmat(mat_file)['traj']  # 存储加载的轨迹数据
        self.T = scp.loadmat(mat_file)['tracks']  # 轨迹跟踪数据
        self.t_h = t_h  # 定义模型在预测时考虑的历史轨迹长度
        self.t_f = t_f  # 定义模型预测未来轨迹的时间长度
        self.d_s = d_s  # 采样数据点，减少计算量
        self.enc_size = enc_size  # size of encoder LSTM隐藏层的维度
        self.grid_size = grid_size  # size of social context grid 社会上下文网格的大小，用于分析车辆周围的环境和邻近车辆
        self.alltime = 0
        self.count = 0

    def __len__(self):  # 查询对象的长度
        return len(self.D)

    def __getitem__(self, idx):  # 它允许使用索引来访问实例 self 中的元素

        dsId = self.D[idx, 0].astype(int)  # dataset id 数据集 ID
        vehId = self.D[idx, 1].astype(int)  # agent id 车辆 ID
        t = self.D[idx, 2]  # frame 时间
        grid = self.D[idx, 11:]  # grid id
        neighbors = []  # 相邻车辆的信息
        neighborsva = []  # 相邻车辆的速度
        neighborslane = []  # 相邻车辆的车道
        neighborsclass = []  # 相邻车辆的类别
        neighborsdistance = []  # 目标车辆到相邻车辆的距离

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId, t, vehId, dsId)   # 获取车辆 vehId 在数据集 dsId 中时间 t 的历史轨迹
        refdistance = np.zeros_like(hist[:, 0])  # 创建一个与 hist 第一列相同的数组，并将所有元素初始化为 0
        refdistance = refdistance.reshape(len(refdistance), 1)  # 将refdistance数组重塑为一个二维数组，其中每行只有一个元素
        fut = self.getFuture(vehId, t, dsId)  # 获取车辆 vehId 在数据集 dsId 中时间 t 的未来轨迹
        va = self.getVA(vehId, t, vehId, dsId)  # 调用 getVA 方法来获取车辆 vehId 在数据集 dsId 中时间 t 的速度和加速度
        lane = self.getLane(vehId, t, vehId, dsId)  # 调用 getLane 方法来获取车辆 vehId 在数据集 dsId 中时间 t 的车道信息
        cclass = self.getClass(vehId, t, vehId, dsId)  # 获取车辆 vehId 在数据集 dsId 中时间 t 的类别信息

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            nbrsdis = self.getHistory(i.astype(int), t, vehId, dsId)  # 获取当前元素 i（转换为整数）的历史轨迹
            if nbrsdis.shape != (0, 2):  # 检查 nbrsdis 的形状是否不是空数组（即至少有一个元素，且每个元素有两个坐标）
                uu = np.power(hist - nbrsdis, 2)  # 计算历史轨迹 hist 和 nbrsdis 之间的每个坐标差的平方
                distancexxx = np.sqrt(uu[:, 0] + uu[:, 1])  # 计算上述坐标差的平方和的平方根，得到距离
                distancexxx = distancexxx.reshape(len(distancexxx), 1)  # 将距离数组 distancexxx 重塑为二维数组，每行只有一个元素
            else:
                distancexxx = np.empty([0, 1])  # 创建一个空的二维数组
            neighbors.append(nbrsdis)  # 当前元素的历史轨迹添加到 neighbors 列表中
            neighborsva.append(self.getVA(i.astype(int), t, vehId, dsId))  # 获取当前元素的速度和加速度信息，并添加到 neighborsva 列表中
            neighborslane.append(self.getLane(i.astype(int), t, vehId, dsId).reshape(-1, 1))   # 获取元素的车道信息并添加
            neighborsclass.append(self.getClass(i.astype(int), t, vehId, dsId).reshape(-1, 1))  # 获取元素的类别信息并添加
            neighborsdistance.append(distancexxx)  # 将计算得到的距离添加到列表中
        lon_enc = np.zeros([args['lon_length']])  # 创建一个长度为 args['lon_length'] 的零数组，用于编码经度信息
        lon_enc[int(self.D[idx, 10] - 1)] = 1  # 将对应索引的元素设置为 1
        lat_enc = np.zeros([args['lat_length']])  # 创建一个长度为 args['lat_length'] 的零数组，用于编码纬度信息
        lat_enc[int(self.D[idx, 9] - 1)] = 1

        # hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

        return hist, fut, neighbors, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance, neighborsdistance, cclass, neighborsclass

    def getLane(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:  # vehId 超出界限
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  # 获取参考车辆在指定数据集和时间的历史轨迹，并转置
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  # 获取目标车辆的历史轨迹，并转置
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 5]  # 找到参考轨迹中时间等于 t 的位置，并获取该位置的第5列的值

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)  # 计算历史轨迹的起始点索引
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1  # 计算历史轨迹的结束点索引
                hist = vehTrack[stpt:enpt:self.d_s, 5]  # 根据起始点、结束点和步长 self.d_s 从目标轨迹中提取第5列的数据

            if len(hist) < self.t_h // self.d_s + 1:  # 提取的历史轨迹长度是否小于预期的最小长度
                return np.empty([0, 1])
            return hist

    def getClass(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  # 获取参考车辆的历史轨迹，并转置
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  # 获取目标车辆的历史轨迹，并转置
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 6]  # 找到参考轨迹中时间等于 t 的位置，并获取该位置的第6列的值

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:  # 检查目标轨迹是否为空或者在时间 t 处没有数据
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)  # 计算历史轨迹的起始点索引
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1  # 计算历史轨迹的结束点索引
                hist = vehTrack[stpt:enpt:self.d_s, 6]  # 找到参考轨迹中时间等于 t 的位置，并获取该位置的第6列的值

            if len(hist) < self.t_h // self.d_s + 1:  # 提取的历史轨迹长度是否小于预期的最小长度
                return np.empty([0, 1])
            return hist

    def getVA(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  # 获取参考车辆的历史轨迹，并转置
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  # 获取目标车辆的历史轨迹，并转置
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 3:5]  # 找到参考轨迹中时间等于 t 的位置，并获取该位置的第3列到第5列的值

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:  # 检查目标轨迹是否为空或者在时间 t 处没有数据
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 3:5]  # 据起始点、结束点和步长 self.d_s 从目标轨迹中提取第3列到第5列的数据

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    # Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):  # 获取指定车辆在特定时间点相对于参考车辆的历史轨迹
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose() 
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            x = np.where(refTrack[:, 0] == t)
            refPos = refTrack[x][0, 1:3]  # 获取参考轨迹中时间 t 的位置信息 第一列到第三列

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h) 
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos  
            if len(hist) < self.t_h // self.d_s + 1:  
                return np.empty([0, 2])
            return hist

    def getdistance(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  # 获取参考车辆的历史轨迹，并转置
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  # 获取目标车辆的历史轨迹，并转置
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos  # 提取目标轨迹中与时间 t 相关的部分，并从参考位置 refPos 中减去，得到相对于参考车辆的位置变化
                hist_ref = refTrack[stpt:enpt:self.d_s, 1:3] - refPos  # 同样提取参考轨迹中与时间 t 相关的部分，并从参考位置 refPos 中减去
                uu = np.power(hist - hist_ref, 2)  # 计算目标轨迹和参考轨迹之间每个点的差的平方
                distance = np.sqrt(uu[:, 0] + uu[:, 1])  # 计算差的平方和的平方根，得到欧几里得距离
                distance = distance.reshape(len(distance), 1)  # 距离数组重塑为列向量

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return distance

    # Helper function to get track future
    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]  # 找到轨迹中时间等于 t 的位置，并获取该位置的第1列到第3列的值 x和y
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos  # 提取从当前时间点 t 起的未来轨迹，并从参考位置 refPos 中减去，得到相对于当前位置的轨迹变化
        return fut

    # Collate function for dataloader
    def collate_fn(self, samples):  # samples 是一个包含多个数据样本的列表
        ttt = time.time()  # 记录当前时间
        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0  # 初始化一个变量来计算批次中非空邻居的数量
        for _, _, nbrs, _, _, _, _, _, _, _, _, _, _ in samples:  # 样本被解构为多个变量，但只有 nbrs（邻居信息）被用于计算非空邻居的数量
            temp = sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])  # 计算列表 nbrs 中非空元素的数量
            nbr_batch_size += temp
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, 2)  # 存储邻居的 x, y 坐标
        nbrsva_batch = torch.zeros(maxlen, nbr_batch_size, 2)  # 存储邻居的速度和加速度信息
        nbrslane_batch = torch.zeros(maxlen, nbr_batch_size, 1)  # 存储邻居的车道信息
        nbrsclass_batch = torch.zeros(maxlen, nbr_batch_size, 1)  # 存储邻居的类别信息
        nbrsdis_batch = torch.zeros(maxlen, nbr_batch_size, 1)  # 存储邻居的距离信息

        # Initialize social mask batch:
        pos = [0, 0]  # 初始化一个名为 pos 的列表，包含两个元素储存x和y
        # 创建一个形状为 (len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)用于存储批次中每个样本的掩码信息
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)  # (batch,3,13,h)
        map_position = torch.zeros(0, 2)  # 创建一个形状为 (0, 2) 的 PyTorch 张量用于存储地图上的位置信息
        mask_batch = mask_batch.bool()  # 将 mask_batch 张量的数据类型转换为布尔型 这通常用于创建掩码

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:

        # 初始化一个形状为 (maxlen, batch_size, 2) 的张量，用于存储每个样本的历史轨迹，其中 batch_size 是 samples 的长度，2 表示每个轨迹点的 x 和 y 坐标
        hist_batch = torch.zeros(maxlen, len(samples), 2)
        # 初始化一个形状为 (maxlen, batch_size, 1) 的张量，用于存储与每个样本相关的历史轨迹点的距离信息
        distance_batch = torch.zeros(maxlen, len(samples), 1)
        # 初始化一个形状为 (len2, batch_size, 2) 的张量，用于存储每个样本的未来轨迹，其中 len2 是基于未来时间范围 self.t_f 和采样步长 self.d_s 计算出的未来轨迹长度
        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2) 用于存储未来轨迹的输出掩码
        lat_enc_batch = torch.zeros(len(samples), args['lat_length'])  # (batch,2) 存储每个样本的横向编码信息
        lon_enc_batch = torch.zeros(len(samples), args['lon_length'])  # (batch,2) 存储每个样本的纵向编码信息
        va_batch = torch.zeros(maxlen, len(samples), 2)  # (maxlen, batch_size, 2) 存储每个样本的速度和加速度信息
        lane_batch = torch.zeros(maxlen, len(samples), 1)  # (maxlen, batch_size, 1) 存储每个样本的车道信息
        class_batch = torch.zeros(maxlen, len(samples), 1)  # (maxlen, batch_size, 1) 的张量，用于存储每个样本的类别信息
        count = 0  # 计数器变量
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for sampleId, (hist, fut, nbrs, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance,
                       neighborsdistance, cclass, neighborsclass) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])  # 将历史轨迹的x坐标转换为张量，并填充到 hist_batch 的对应位置
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])  # 将历史轨迹的y坐标转换为张量，并填充到 hist_batch 的对应位置
            distance_batch[0:len(hist), sampleId, :] = torch.from_numpy(refdistance)  # 参考距离数据填充到 distance_batch
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])  # 将未来轨迹的 x 和 y 坐标数据填充到 fut_batch
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut), sampleId, :] = 1  # 为未来轨迹数据设置一个操作掩码
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)  # 将横向和纵向编码数据转换为 PyTorch 张量并填充到对应的批次中
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            va_batch[0:len(va), sampleId, 0] = torch.from_numpy(va[:, 0])  # 将速度和加速度数据填充到 va_batch
            va_batch[0:len(va), sampleId, 1] = torch.from_numpy(va[:, 1])
            lane_batch[0:len(lane), sampleId, 0] = torch.from_numpy(lane)  # 车道数据填充到 lane_batch
            class_batch[0:len(cclass), sampleId, 0] = torch.from_numpy(cclass)  # 类别数据填充到 class_batch
            # Set up neighbor, neighbor sequence length, and mask batches:
            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_batch[0:len(nbr), count, 0] = torch.from_numpy(nbr[:, 0])  # 如果数据非空，其x坐标转换为张量填充到nbrs_batch的对应位置
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])  # 如果数据非空，其y坐标转换为张量填充到nbrs_batch的对应位置
                    pos[0] = id % self.grid_size[0]  # 邻居在网格中的 x 位置，使用模运算
                    pos[1] = id // self.grid_size[0]  # 计算邻居在网格中的 y 位置，使用整除运算
                    # 对于当前样本和邻居的网格位置，设置一个全 1 的掩码，掩码大小为 self.enc_size，数据类型为字节（byte）
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(self.enc_size).byte()
                    # 将邻居的网格位置添加到 map_position 张量中
                    map_position = torch.cat((map_position, torch.tensor([[pos[1], pos[0]]])), 0)
                    count += 1  # 更新邻居计数器
            for id, nbrva in enumerate(neighborsva):  # 循环遍历 neighborsva
                if len(nbrva) != 0:
                    nbrsva_batch[0:len(nbrva), count1, 0] = torch.from_numpy(nbrva[:, 0])  # 将其速度转换张量并填充到 nbrsva_batch
                    nbrsva_batch[0:len(nbrva), count1, 1] = torch.from_numpy(nbrva[:, 1])  # 将其加速度转换张量并填充到 nbrsva_batch
                    count1 += 1  # 更新邻居的速度和加速度计数器

            # for id, nbrlane in enumerate(neighborslane):
            #     if len(nbrlane) != 0:
            #         for nbrslanet in range(len(nbrlane)):
            #             nbrslane_batch[nbrslanet, count2, int(nbrlane[nbrslanet] - 1)] = 1
            #         count2 += 1
            for id, nbrlane in enumerate(neighborslane):
                if len(nbrlane) != 0:  # 将当前邻居的车道信息从 NumPy 数组转换为 PyTorch 张量，并填充到 nbrslane_batch
                    nbrslane_batch[0:len(nbrlane), count2, :] = torch.from_numpy(nbrlane)
                    count2 += 1

            for id, nbrdis in enumerate(neighborsdistance):
                if len(nbrdis) != 0:  # 将邻居的距离信息填充到 nbrsdis_batch
                    nbrsdis_batch[0:len(nbrdis), count3, :] = torch.from_numpy(nbrdis)
                    count3 += 1

            for id, nbrclass in enumerate(neighborsclass):
                if len(nbrclass) != 0:  # 将邻居的类别信息填充到 nbrsclass_batch
                    nbrsclass_batch[0:len(nbrclass), count4, :] = torch.from_numpy(nbrclass)
                    count4 += 1
        #  mask_batch 
        self.alltime += (time.time() - ttt)  # 计算数据加载操作的总耗时
        self.count += args['num_worker']  # 表示增加工作进程的数量
        # if (self.count > args['time']):
        #    print(self.alltime / self.count, "data load time")
        # 构成了模型训练或推理的输入批次，包含了所需的所有信息，例如轨迹、邻居信息、编码信息、掩码 collate_fn 函数将多个样本的数据整合成批次数据，使其准备好被用于深度学习模型
        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch, va_batch, nbrsva_batch, lane_batch, nbrslane_batch, distance_batch, nbrsdis_batch, class_batch, nbrsclass_batch, map_position

class HighdDataset(Dataset):

    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3)):
        self.D = np.transpose(h5py.File(mat_file, 'r')['traj'].value)
        self.T = h5py.File(mat_file, 'r')
        ref = self.T['tracks'][0][0]
        res = self.T[ref]
        self.t_h = t_h  # 
        self.t_f = t_f  # 
        self.d_s = d_s  # skip
        self.enc_size = enc_size  # size of encoder LSTM
        self.grid_size = grid_size  # size of social context grid

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):

        dsId = self.D[0, idx].astype(int)  
        vehId = self.D[1, idx].astype(int)  
        t = self.D[2, idx] 
        grid = self.D[14:, idx]  
        neighbors = []
        neighborsva = []
        neighborslane = []
        neighborsclass = []
        neighborsdistance = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId, t, vehId, dsId)  
        refdistance = np.zeros_like(hist[:, 0])
        refdistance = refdistance.reshape(len(refdistance), 1)
        fut = self.getFuture(vehId, t, dsId)  
        va = self.getVA(vehId, t, vehId, dsId)
        lane = self.getLane(vehId, t, vehId, dsId)
        cclass = self.getClass(vehId, t, vehId, dsId)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            nbrsdis = self.getHistory(i.astype(int), t, vehId, dsId)
            if nbrsdis.shape != (0, 2):
                uu = np.power(hist - nbrsdis, 2)
                distancexxx = np.sqrt(uu[:, 0] + uu[:, 1])
                distancexxx = distancexxx.reshape(len(distancexxx), 1)
            else:
                distancexxx = np.empty([0, 1])
            neighbors.append(nbrsdis) 
            neighborsva.append(self.getVA(i.astype(int), t, vehId, dsId))
            neighborslane.append(self.getLane(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsclass.append(self.getClass(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsdistance.append(distancexxx)
        lon_enc = np.zeros([2])
        lon_enc[int(self.D[idx, 13] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 12] - 1)] = 1

        # hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

        return hist, fut, neighbors, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance, neighborsdistance, cclass, neighborsclass

    def getLane(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 8] 

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 8]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getClass(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 5]  

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getVA(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 6:8] 

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 6:8] - refPos

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    # Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            x = np.where(refTrack[:, 0] == t)
            refPos = refTrack[x][0, 1:3]  

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)  
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1 
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos 
            if len(hist) < self.t_h // self.d_s + 1:  
                return np.empty([0, 2])
            return hist

    def getdistance(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3] 

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
                hist_ref = refTrack[stpt:enpt:self.d_s, 1:3] - refPos
                uu = np.power(hist - hist_ref, 2)
                distance = np.sqrt(uu[:, 0] + uu[:, 1])
                distance = distance.reshape(len(distance), 1)

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return distance

    # Helper function to get track future
    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3] 
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut

    # Collate function for dataloader
    def collate_fn(self, samples):
        nowt = time.time()
        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _, _, nbrs, _, _, _, _, _, _, _, _, _, _ in samples:
            temp = sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
            nbr_batch_size += temp
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, 2)  # (len,batch*车数，2)
        nbrsva_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrslane_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsclass_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsdis_batch = torch.zeros(maxlen, nbr_batch_size, 1)

        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)  # (batch,3,13,h)
        map_position = torch.zeros(0, 2)
        mask_batch = mask_batch.bool()

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen, len(samples), 2)  # (len1,batch,2)
        distance_batch = torch.zeros(maxlen, len(samples), 1)
        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
        lat_enc_batch = torch.zeros(len(samples), 3)  # (batch,3)
        lon_enc_batch = torch.zeros(len(samples), 2)  # (batch,2)
        va_batch = torch.zeros(maxlen, len(samples), 2)
        lane_batch = torch.zeros(maxlen, len(samples), 1)
        class_batch = torch.zeros(maxlen, len(samples), 1)
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for sampleId, (hist, fut, nbrs, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance,
                       neighborsdistance, cclass, neighborsclass) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            distance_batch[0:len(hist), sampleId, :] = torch.from_numpy(refdistance)
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut), sampleId, :] = 1
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            va_batch[0:len(va), sampleId, 0] = torch.from_numpy(va[:, 0])
            va_batch[0:len(va), sampleId, 1] = torch.from_numpy(va[:, 1])
            lane_batch[0:len(lane), sampleId, 0] = torch.from_numpy(lane)
            class_batch[0:len(cclass), sampleId, 0] = torch.from_numpy(cclass)
            # Set up neighbor, neighbor sequence length, and mask batches:
            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_batch[0:len(nbr), count, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(self.enc_size).byte()
                    map_position = torch.cat((map_position, torch.tensor([[pos[1], pos[0]]])), 0)
                    count += 1
            for id, nbrva in enumerate(neighborsva):
                if len(nbrva) != 0:
                    nbrsva_batch[0:len(nbrva), count1, 0] = torch.from_numpy(nbrva[:, 0])
                    nbrsva_batch[0:len(nbrva), count1, 1] = torch.from_numpy(nbrva[:, 1])
                    count1 += 1

            # for id, nbrlane in enumerate(neighborslane):
            #     if len(nbrlane) != 0:
            #         for nbrslanet in range(len(nbrlane)):
            #             nbrslane_batch[nbrslanet, count2, int(nbrlane[nbrslanet] - 1)] = 1
            #         count2 += 1
            for id, nbrlane in enumerate(neighborslane):
                if len(nbrlane) != 0:
                    nbrslane_batch[0:len(nbrlane), count2, :] = torch.from_numpy(nbrlane)
                    count2 += 1

            for id, nbrdis in enumerate(neighborsdistance):
                if len(nbrdis) != 0:
                    nbrsdis_batch[0:len(nbrdis), count3, :] = torch.from_numpy(nbrdis)
                    count3 += 1

            for id, nbrclass in enumerate(neighborsclass):
                if len(nbrclass) != 0:
                    nbrsclass_batch[0:len(nbrclass), count4, :] = torch.from_numpy(nbrclass)
                    count4 += 1
        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch, va_batch, nbrsva_batch, lane_batch, nbrslane_batch, distance_batch, nbrsdis_batch, class_batch, nbrsclass_batch, map_position
