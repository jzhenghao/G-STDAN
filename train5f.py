from torch.utils.data import DataLoader
import loader2 as lo
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import torch
import datetime
import os
from evaluate5f import Evaluate
from config import *
import multiprocessing
import csv


def maskedNLL(y_pred, y_gt, mask):  # 计算一个自定义的负对数似然损失
    # mask = t.cat((mask[:15, :, :], t.zeros_like(mask[15:, :, :])), dim=0)
    acc = t.zeros_like(mask)  # 累积损失值
    muX = y_pred[:, :, 0]  # 提取均值（muX, muY）、标准差（sigX, sigY）和相关系数（rho）
    muY = y_pred[:, :, 1]
    sigX = y_pred[:, :, 2]
    sigY = y_pred[:, :, 3]
    rho = y_pred[:, :, 4]
    ohr = t.pow(1 - t.pow(rho, 2), -0.5)  # (1-rhp^2)^0.5 协方差矩阵
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    # If we represent likelihood in feet^(-1)
    out = 0.5 * t.pow(ohr, 2) * (
            t.pow(sigX, 2) * t.pow(x - muX, 2) + t.pow(sigY, 2) * t.pow(y - muY, 2) - 2 * rho * t.pow(sigX,
                                                                                                      1) * t.pow(
        sigY, 1) * (x - muX) * (y - muY)) - t.log(sigX * sigY * ohr) + 1.8379
    #  m^(-1):meter out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX,
    # 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX)
    # * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
    acc[:, :, 0] = out  # 对数似然赋值给acc张量的两个通道
    acc[:, :, 1] = out
    acc = acc * mask  # 累积损失张量acc与掩码mask相乘忽略不需要计算损失的部分
    lossVal = t.sum(acc) / t.sum(mask)  # 得到平均损失
    return lossVal


def MSELoss2(g_out, fut, mask):  # 均方误差损失
    acc = t.zeros_like(mask)  # 创建一个与 mask 形状相同的零张量 acc，用于累加误差
    muX = g_out[:, :, 0]  # 预测的 x 坐标
    muY = g_out[:, :, 1]  # 预测的 y 坐标
    x = fut[:, :, 0]  # 实际的 x 坐标
    y = fut[:, :, 1]  # 实际的 y 坐标
    out = t.pow(x - muX, 2) + t.pow(y - muY, 2)  # 计算预测值和实际值之间的平方差并相加
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask  # 指定哪些元素在计算损失时应该被考虑
    lossVal = t.sum(acc) / t.sum(mask)  # 最终计算出的损失值
    return lossVal


def CELoss(pred, target):  # pred是模型预测的概率分布target是实际的标签
    value = t.log(t.sum(pred * target, dim=-1))  # t.log对求和的结果取自然对数
    return -t.sum(value) / value.shape[0]  # 平均交叉熵损失
def get_mix_weight(epoch, pre_epoch, decay_epochs=20):
    if epoch < pre_epoch:
        return 1.0
    else:
        progress = min(1.0, (epoch - pre_epoch) / decay_epochs)
        return max(0.1, 1.0 - 0.9 * progress)

def main():
    args['train_flag'] = True
    evaluate = Evaluate()

    gdEncoder = model.GDEncoder(args)
    generator = model.Generator(args)

    gdEncoder = gdEncoder.to(device)
    generator = generator.to(device)
    gdEncoder.train()  # 将gdEncoder模型设置为训练模式
    generator.train()
    if dataset == "ngsim":
        # 直接使用完整数据集，不再创建子集
        if args['lon_length'] == 3:
            t1 = lo.NgsimDataset(r'data/SampledTrainSet.mat')
        else:
            t1 = lo.NgsimDataset(r'data/SampledTrainSet.mat')



        print(f"Using full NGSIM dataset with {len(t1)} samples")
        trainDataloader = DataLoader(
            t1,
            batch_size=args['batch_size'],
            shuffle=True,
            collate_fn=t1.collate_fn,  # 注意：原始完整数据集应直接使用collate_fn
            pin_memory=True
        )
    else:
        t1 = lo.HighdDataset(r'data/SampledTrainSet.mat')
        trainDataloader = DataLoader(
            t1,
            batch_size=args['batch_size'],
            shuffle=True,
            collate_fn=t1.collate_fn,
            pin_memory=True
        )

    # 优化器和学习率调度器保持不变
    optimizer_gd = optim.Adam(gdEncoder.parameters(), lr=args['learning_rate'])
    optimizer_g = optim.Adam(generator.parameters(), lr=args['learning_rate'])
    scheduler_gd = ExponentialLR(optimizer_gd, gamma=0.6)
    scheduler_g = ExponentialLR(optimizer_g, gamma=0.6)

    # --- 添加损失历史记录文件处理 ---
    loss_history_file = 'loss_history.csv'
    # Check if file exists and write header only if it's a new file
    if not os.path.exists(loss_history_file):
        with open(loss_history_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss'])
    # Open file in append mode for writing loss data
    loss_file = open(loss_history_file, 'a', newline='')
    loss_writer = csv.writer(loss_file)
    # ---------------------------------

    for epoch in range(args['epoch']):
        start_time = datetime.datetime.now()
        print(f"Epoch {epoch + 1} started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("epoch:", epoch + 1, 'lr', optimizer_g.param_groups[0]['lr'])

        # --- 添加 epoch 损失累加变量 ---
        epoch_loss_sum = 0.0
        # ----------------------------
        loss_gi1 = 0  # 计算损失值
        loss_gix = 0
        loss_gx_2i = 0
        loss_gx_3i = 0

        for idx, data in enumerate(tqdm(trainDataloader)):
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions = data
            hist = hist.to(device)  # 移动到device
            nbrs = nbrs.to(device)
            mask = mask.to(device)
            lat_enc = lat_enc.to(device)
            lon_enc = lon_enc.to(device)
            fut = fut[:args['out_length'], :, :]  # 从fut变量中选择前args['out_length']个时间步的数据
            fut = fut.to(device)
            op_mask = op_mask[:args['out_length'], :, :]  # 从op_mask变量中选择前args['out_length']个时间步的数据
            op_mask = op_mask.to(device)
            va = va.to(device)
            nbrsva = nbrsva.to(device)
            lane = lane.to(device)
            nbrslane = nbrslane.to(device)
            dis = dis.to(device)
            nbrsdis = nbrsdis.to(device)
            map_positions = map_positions.to(device)
            cls = cls.to(device)
            nbrscls = nbrscls.to(device)
            # 调用gdEncoder函数，将多个输入变量编码为values
            values = gdEncoder(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls)
            # 调用generator函数，使用编码后的values和编码数据生成输出，包括g_out（可能代表生成的输出），以及预测的纬度和经度
            g_out, lat_pred, lon_pred = generator(values, lat_enc, lon_enc)
            # 修改后的代码段
            if args['use_mse']:
                loss_g1 = MSELoss2(g_out, fut, op_mask)
            else:
                mse_weight = get_mix_weight(epoch, args['pre_epoch'])
                loss_g1 = (
                        mse_weight * MSELoss2(g_out, fut, op_mask) +
                        (1 - mse_weight) * maskedNLL(g_out, fut, op_mask)
                )
                # 可选：打印当前混合比例（调试用）
                if idx % 10000 == 0:
                    print(f'Epoch {epoch}, MSE Weight: {mse_weight:.2f}')
            loss_gx_3 = CELoss(lat_pred, lat_enc)  # 计算纬度预测值lat_pred和实际编码值lat_enc之间的交叉熵损失
            loss_gx_2 = CELoss(lon_pred, lon_enc)  # 计算经度预测值lon_pred和实际编码值lon_enc之间的交叉熵损失
            loss_gx = loss_gx_3 + loss_gx_2  # 地理坐标预测的总损失
            loss_g = loss_g1 + 1 * loss_gx  # 得到生成器的总损失
            optimizer_g.zero_grad()  # 清除optimizer_g和optimizer_gd优化器中的梯度，为反向传播准备
            optimizer_gd.zero_grad()
            loss_g.backward()  # 生成器的总损失进行反向传播，计算梯度
            a = t.nn.utils.clip_grad_norm_(generator.parameters(), 10)  # 对生成器generator的参数梯度进行裁剪，确保梯度的最大范数不超过10
            a = t.nn.utils.clip_grad_norm_(gdEncoder.parameters(), 10)
            optimizer_g.step()  # 根据计算出的梯度更新生成器generator和地理编码器gdEncoder的参数
            optimizer_gd.step()

            # --- 累加 epoch 损失 ---
            epoch_loss_sum += loss_g.item()
            # ----------------------

            loss_gi1 += loss_g1.item()  # loss_gi1用于累积loss_g1的损失值
            loss_gx_2i += loss_gx_2.item()  # 用于累积loss_gx_2的损失值
            loss_gx_3i += loss_gx_3.item()  # 用于累积loss_gx_3的损失值
            loss_gix += loss_gx.item()  # 用于累积loss_gx的总损失值
            if idx % 100000 == 9999:
                print('mse:', loss_gi1 / 100000, '|loss_gx_2:', loss_gx_2i / 100000, '|loss_gx_3', loss_gx_3i / 100000)
                print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                loss_gi1 = 0  # 重置为0为下一次迭代做准备
                loss_gix = 0
                loss_gx_2i = 0
                loss_gx_3i = 0

        # --- Epoch 结束时保存损失 ---
        avg_epoch_loss = epoch_loss_sum / len(trainDataloader)
        loss_writer.writerow([epoch + 1, avg_epoch_loss])
        loss_file.flush() # 确保数据写入文件
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}")
        # ---------------------------

        save_model(name=str(epoch + 1), gdEncoder=gdEncoder,
                   generator=generator, path=args['path'])
        evaluate.main(name=str(epoch + 1), val=True)  # 调用evaluate模块的main函数来评估模型的性能
        scheduler_gd.step()  # 调用scheduler_gd对象的step方法来更新编码器gdEncoder的学习率
        scheduler_g.step()  # 调用scheduler_g对象的step方法来更新生成器generator的学习率

    # --- 关闭损失历史文件 ---
    loss_file.close()
    # -----------------------


def save_model(name, gdEncoder, generator, path):
    l_path = args['path']
    if not os.path.exists(l_path):
        os.makedirs(l_path)
    t.save(gdEncoder.state_dict(), l_path + '/epoch' + name + '_gd.tar')  # save 函数保存 gdEncoder 模型的参数
    t.save(generator.state_dict(), l_path + '/epoch' + name + '_g.tar')  # save 函数保存 generator 模型的参数


# if __name__ == '__main__':
if __name__ == '__main__':
    main()