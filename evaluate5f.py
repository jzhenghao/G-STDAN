from __future__ import print_function
import loader2 as lo
from torch.utils.data import DataLoader
from config import *
import os
import time
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'





class Evaluate():

    def __init__(self):
        self.op = 0  # 实例变量
        self.scale = 0.3048  # 比例因子
        self.prop = 1

    def maskedMSETest(self, y_pred, y_gt, mask):
        acc = t.zeros_like(mask)  # 创建一个与 mask 形状相同的张量，初始化为0
        muX = y_pred[:, :, 0]  # 从 y_pred 中提取预测的X和Y坐标
        muY = y_pred[:, :, 1]
        x = y_gt[:, :, 0]  # y_gt 中提取真实的X和Y坐标
        y = y_gt[:, :, 1]
        out = t.pow(x - muX, 2) + t.pow(y - muY, 2)  # 计算预测坐标和真实坐标之间的平方差这是均方误差的计算方式
        acc[:, :, 0] = out  # 将计算出的平方差赋值给 acc 的第一个和第二个通道
        acc[:, :, 1] = out
        acc = acc * mask  # 为了在评估中只考虑掩码区域
        lossVal = t.sum(acc[:, :, 0], dim=1)  # 计算每个样本在第一个通道上的累积误差
        counts = t.sum(mask[:, :, 0], dim=1)  # 每个样本在掩码区域中的有效像素数量
        loss = t.sum(acc) / t.sum(mask)  # 计算整体的损失值，这是所有样本累积误差与有效像素数量的比值
        return lossVal, counts, loss

    # Helper function for log sum exp calculation: 一个计算公式
    def logsumexp(self, inputs, dim=None, keepdim=False):  # 一个在深度学习中常用的操作，用于计算一组数的指数加权和的对数
        if dim is None:  # 通过减去最大值来减少计算指数时的数值范围，从而避免在对数变换之前发生数值溢出
            inputs = inputs.view(-1)
            dim = 0  # 将inputs变换为一维张量，并设置dim为0
        s, _ = t.max(inputs, dim=dim, keepdim=True)
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
        if not keepdim:
            outputs = outputs.squeeze(dim)
        return outputs

    def maskedNLLTest(self, fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes=2,
                      use_maneuvers=True):  # 计算一个定制的负对数似然损失
        if use_maneuvers:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).to(device)
            count = 0  # 初始化一个张量acc来累加损失以及一个计数器count
            for k in range(num_lon_classes):
                for l in range(num_lat_classes):
                    wts = lat_pred[:, l] * lon_pred[:, k]
                    # 计算每个纬度和经度类别组合的权重，纬度类别概率 lat_pred 的第l个元素和经度类别概率 lon_pred 的第k个元素的乘积
                    wts = wts.repeat(len(fut_pred[0]), 1)  # 将权重wts沿第一个维度重复，以匹配fut_pred中第一个元素的长度
                    y_pred = fut_pred[k * num_lat_classes + l]  # 从fut_pred中选择当前纬度和经度类别组合对应的预测数据
                    y_gt = fut  # 将实际的未来位置数据赋值给y_gt
                    muX = y_pred[:, :, 0]  # 提取第一个维度 X坐标的均值
                    muY = y_pred[:, :, 1]  # 提取第二个维度 y坐标的均值
                    sigX = y_pred[:, :, 2]  # 提取第三个维度 X坐标的标准差
                    sigY = y_pred[:, :, 3]  # 提取第四个维度 y坐标的标准差
                    rho = y_pred[:, :, 4]  # 提取第五个维度 代表X和Y坐标的相关系数
                    ohr = t.pow(1 - t.pow(rho, 2), -0.5)  # 计算rho的反正切函数，这是用于计算双变量高斯分布的协方差矩阵的一部分
                    x = y_gt[:, :, 0]  # 从y_gt中提取X坐标
                    y = y_gt[:, :, 1]  # 从y_gt中提取y坐标
                    # If we represent likelihood in feet^(-1):
                    out = -(0.5 * t.pow(ohr, 2) * (  # 计算负对数似然损失的数值部分
                            t.pow(sigX, 2) * t.pow(x - muX, 2) + 0.5 * t.pow(sigY, 2) * t.pow(
                        y - muY, 2) - rho * t.pow(sigX, 1) * t.pow(sigY, 1) * (x - muX) * (
                                    y - muY)) - t.log(sigX * sigY * ohr) + 1.8379)
                    acc[:, :, count] = out + t.log(wts)  # 将计算出的负对数似然损失加上权重wts的对数，存储在累加器acc的相应位置
                    count += 1
            acc = -self.logsumexp(acc, dim=2)  # 应用logsumexp沿着第三个维度（dim=2）对acc进行操作，一种数值稳定的对数求和操作，处理潜在的数值下溢问题
            acc = acc * op_mask[:, :, 0]  # 屏蔽某些不需要计算损失的预测
            loss = t.sum(acc) / t.sum(op_mask[:, :, 0])  # 计算总损失
            lossVal = t.sum(acc, dim=1)  # 计算每个样本的损失值
            counts = t.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts, loss
        else:  # 创建一个形状为(op_mask.shape[0], op_mask.shape[1], 1)的零张量acc
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], 1).to(device)
            y_pred = fut_pred  # 将预测的未来位置数据赋值给y_pred
            y_gt = fut  # 将实际的未来位置数据赋值给y_gt
            muX = y_pred[:, :, 0]  # 提取x坐标的均值
            muY = y_pred[:, :, 1]  # 提取Y坐标的均值
            sigX = y_pred[:, :, 2]  # 提取X坐标的标准差
            sigY = y_pred[:, :, 3]  # 提取y坐标的标准差
            rho = y_pred[:, :, 4]  # 提取相关系数
            ohr = t.pow(1 - t.pow(rho, 2), -0.5)  # 计算相关系数的反正切函数的逆，用于计算双变量高斯分布的协方差项
            x = y_gt[:, :, 0]  # 提取实际X坐标
            y = y_gt[:, :, 1]  # 提取实际y坐标
            # If we represent likelihood in feet^(-1):
            out = 0.5 * t.pow(ohr, 2) * (
                    t.pow(sigX, 2) * t.pow(x - muX, 2) + t.pow(sigY, 2) * t.pow(y - muY,
                                                                                2) - 2 * rho * t.pow(
                sigX, 1) * t.pow(sigY, 1) * (x - muX) * (y - muY)) - t.log(sigX * sigY * ohr) + 1.8379  # 计算负对数似然损失的数值部分
            acc[:, :, 0] = out  # 将计算出的负对数似然损失存储在累加器acc的第一个通道
            acc = acc * op_mask[:, :, 0:1]  # 屏蔽某些不需要计算损失的预测
            loss = t.sum(acc[:, :, 0]) / t.sum(op_mask[:, :, 0])  # 计算总损失
            lossVal = t.sum(acc[:, :, 0], dim=1)  # 计算每个样本的损失值
            counts = t.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts, loss
    def main(self, name, val):
        model_step = 1  # 初始化一个变量 model_step 并设置为 1
        # args['train_flag'] = not args['use_maneuvers']
        args['train_flag'] = True  # 表示模型将处于训练模式
        l_path = args['path']  # 从 args 字典中获取路径，并将其存储在变量 l_path 中
        generator = model.Generator(args=args)  # 创建 Generator 类的一个实例
        gdEncoder = model.GDEncoder(args=args)  # 创建 GDEncoder 类的一个实例
        # 加载预训练的生成器模型的状态字典，文件路径由 l_path、name 和 '_g.tar' 组成
        generator.load_state_dict(t.load(l_path + '/epoch' + name + '_g.tar', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        # 加载预训练的GD编码器模型的状态字典，文件路径由 l_path、name 和 '_gd.tar' 组成
        gdEncoder.load_state_dict(t.load(l_path + '/epoch' + name + '_gd.tar', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        generator = generator.to(device)
        gdEncoder = gdEncoder.to(device)
        generator.eval()  # 将生成器模型设置为评估模式
        gdEncoder.eval()  # 将GD编码器模型设置为评估模式
        if val:
            if dataset == "ngsim":
                if args['lon_length'] == 3:
                    t2 = lo.NgsimDataset('data/SampledValSet.mat')
                else:
                    t2 = lo.NgsimDataset('data/SampledValSet.mat')
            else:
                t2 = lo.HighdDataset('Val')
            print(args['num_worker'])
            valDataloader = DataLoader(t2, batch_size=args['batch_size'], shuffle=True, num_workers=0,
                                       collate_fn=t2.collate_fn)  # 6716batch 从 t2 数据集中批量加载数据
        else:
            if dataset == "ngsim":
                if args['lon_length'] == 3:
                    t2 = lo.NgsimDataset('E:/aaa/stdan-master/stdan-master/data/SampledValSet.mat')
                else:
                    t2 = lo.NgsimDataset('E:/aaa/stdan-master/stdan-master/data/SampledValSet.mat')
            else:
                t2 = lo.HighdDataset('Test')
            print(args['num_worker'])
            valDataloader = DataLoader(t2, batch_size=args['batch_size'], shuffle=True, num_workers=0,
                                       collate_fn=t2.collate_fn)

        lossVals = t.zeros(args['out_length']).to(device)  # 用于存储每个时间步的损失值
        counts = t.zeros(args['out_length']).to(device)
        avg_val_loss = 0  # 存储整个验证过程的平均损失值
        all_time = 0
        nbrsss = 0

        val_batch_count = len(valDataloader)  # 获取验证数据加载器valDataloader中的批次总数
        print("begin.................................", name)
        with(t.no_grad()):  # 评估过程中不会计算梯度，从而减少内存消耗并加快计算速度
            for idx, data in enumerate(valDataloader):
                hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions = data
                hist = hist.to(device)  # 变量移动到指定的设备
                nbrs = nbrs.to(device)
                mask = mask.to(device)
                lat_enc = lat_enc.to(device)
                lon_enc = lon_enc.to(device)
                fut = fut[:args['out_length'], :, :]  # 根据args['out_length']截取fut（未来预测目标）的前几帧
                fut = fut.to(device)
                op_mask = op_mask[:args['out_length'], :, :]  # 确保输入数据的维度与模型期望的输出维度相匹配
                op_mask = op_mask.to(device)
                va = va.to(device)
                nbrsva = nbrsva.to(device)
                lane = lane.to(device)
                nbrslane = nbrslane.to(device)
                cls = cls.to(device)
                nbrscls = nbrscls.to(device)
                map_positions = map_positions.to(device)
                te = time.time()  # 记录当前时间
                values = gdEncoder(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls)  # 传入多个参数生成编码后的值values
                fut_pred, lat_pred, lon_pred = generator(values, lat_enc, lon_enc)
                all_time += time.time() - te  # 评估过程总推理时间


                if not args['train_flag']:
                    indices = []  # 存储索引值
                    if args['val_use_mse']:  # 使用均方误差
                        fut_pred_max = t.zeros_like(fut_pred[0])  # 创建一个与fut_pred[0]形状相同的零张量fut_pred_max
                        for k in range(lat_pred.shape[0]):  # 128
                            lat_man = t.argmax(lat_enc[k, :]).detach()  # 代表了预测的纬度和纬度类别并使用detach()方法从计算图中分离
                            lon_man = t.argmax(lon_enc[k, :]).detach()
                            index = lon_man * 3 + lat_man  # 预测的经度和纬度类别计算出一个索引，这里3是经度类别的数量
                            indices.append(index)  # 将计算出的索引添加到indices列表中
                            fut_pred_max[:, k, :] = fut_pred[index][:, k, :]  # 计算出索引从fut_pred中选择对应的预测结果赋值给fut_pred_max
                        l, c, loss = self.maskedMSETest(fut_pred_max, fut, op_mask)  # 计算MSE损失
                    else:
                        l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                                                        use_maneuvers=args['use_maneuvers'])  # 计算负对数似然损失

                else:
                    if args['val_use_mse']:
                        l, c, loss = self.maskedMSETest(fut_pred, fut, op_mask)  # 使用均方误差（MSE）作为评估指标
                    else:
                        l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                                                        use_maneuvers=args['use_maneuvers'])  # 计算负对数似然损失

                lossVals += l.detach()  # 将当前批次的损失值l累加到lossVals变量上
                counts += c.detach()  # 将当前批次的计数c累加到counts变量上
                avg_val_loss += loss.item()  # 将当前批次的损失值loss转换为一个标量累加到avg_val_loss变量上，用于后续计算平均损失
                if idx == int(val_batch_count / 8) * model_step:
                    print('process:', model_step / 8)  # 证过程中的特定点（四分之一）输出进度信息
                    model_step += 1  # 给出了已经处理的批次占总批次的比例
            if args['val_use_mse']:
                print('valmse:', avg_val_loss / val_batch_count)  # 到目前为止计算的MSE损失值的平均值
                print(t.pow(lossVals / counts, 0.5) * 0.3048)  # 计算均方根误差（RMSE），乘以0.3048将单位从英尺转换为米
            else:
                print('valnll:', avg_val_loss / val_batch_count)  # 计算的负对数似然（NLL）损失值的平均值
                print(lossVals / counts)  # 打印最终的损失值


if __name__ == '__main__':

    names = ['9']
    evaluate = Evaluate()
    for epoch in names:
        evaluate.main(name=epoch, val=True)