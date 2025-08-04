%% 本项目提供了一个基于 PyTorch 的车辆轨迹预测完整方案，融合 BiLSTM 时序编码、GCN 空间交互、多头注意力与意图感知解码，在 NGSIM 和 highD 数据集上实现多模态轨迹生成。仓库包含训练脚本 `train5f.py`（自动混合 MSE/NLL 损失）、评估脚本 `evaluate5f.py`（输出 RMSE 或 NLL）、模型实现 `model5f_mult.py`、数据加载器 `loader2.py` 及配置文件 `config.py`。只需把下载的 `.mat` 数据放入 `data/` 目录，运行 `python train5f.py` 即可开始训练，结果自动保存至 `checkpoint/true_1/`，再执行 `python evaluate5f.py` 查看验证集性能。
## the ngsim dataset
the dataset can download from this 
link：https://pan.baidu.com/s/1ur34Au8h3b3WZFM5a2RWOw (4p44) 
下面是matlab代码 对数据集进行预处理
%% Process dataset into mat files %%

clear;
clc;

%% Inputs:
% Locations of raw input files:
us101_1 = 'E:/matlab/raw-data/trajectories-0500pm-0515pm/trajectories-0500-0515.txt';
us101_2 = 'E:/matlab/raw-data/trajectories-0515pm-0530pm/trajectories-0515-0530.txt';
us101_3 = 'E:/matlab/raw-data/trajectories-0400pm-0415pm/trajectories-0400-0415.txt';
i80_1 = 'E:/matlab/raw-data/trajectories-0750am-0805am/trajectories-0750am-0805am.txt';
i80_2 = 'E:/matlab/raw-data/trajectories-0805am-0820am/trajectories-0805am-0820am.txt';
i80_3 = 'E:/matlab/raw-data/trajectories-0820am-0835am/trajectories-0820am-0835am.txt';

/
%% Fields: 

%{ 
1: Dataset Id
2: Vehicle Id
3: Frame Number
4: Local X
5: Local Y
6: Lane Id
7: Lateral maneuver
8: Longitudinal maneuver
9-47: Neighbor Car Ids at grid location
%}

%% Load data and add dataset id
disp('Loading data...')
traj{1} = load(us101_1);    
traj{1} = single([ones(size(traj{1},1),1),traj{1}]);
traj{2} = load(us101_2);
traj{2} = single([2*ones(size(traj{2},1),1),traj{2}]);
traj{3} = load(us101_3);
traj{3} = single([3*ones(size(traj{3},1),1),traj{3}]);
traj{4} = load(i80_1);    
traj{4} = single([4*ones(size(traj{4},1),1),traj{4}]);
traj{5} = load(i80_2);
traj{5} = single([5*ones(size(traj{5},1),1),traj{5}]);
traj{6} = load(i80_3);
traj{6} = single([6*ones(size(traj{6},1),1),traj{6}]);

for k = 1:6
    traj{k} = traj{k}(:,[1,2,3,6,7,13,14,15,12]);
    if k <=3
        traj{k}(traj{k}(:,8)>=6,8) = 6;
    end
end


%% Parse fields (listed above):
disp('Parsing fields...')

for ii = 1:6
    vehIds = unique(traj{ii}(:,2));

    for v = 1:length(vehIds)
        vehTrajs{ii}(int2str(vehIds(v))) = traj{ii}(traj{ii}(:,2) == vehIds(v),:);
    end
    
    timeFrames = unique(traj{ii}(:,3));

    for v = 1:length(timeFrames)
        vehTimes{ii}(int2str(timeFrames(v))) = traj{ii}(traj{ii}(:,3) == timeFrames(v),:);
    end
    
    for k = 1:length(traj{ii}(:,1))        
        time = traj{ii}(k,3);
        dsId = traj{ii}(k,1);
        vehId = traj{ii}(k,2);
        vehtraj = vehTrajs{ii}(int2str(vehId));
        ind = find(vehtraj(:,3)==time);
        ind = ind(1);
        lane = traj{ii}(k,8);
        
        
       %% Get lateral maneuver:
        ub = min(size(vehtraj,1),ind+40);
        lb = max(1, ind-40);
        if vehtraj(ub,8)>vehtraj(ind,8) || vehtraj(ind,8)>vehtraj(lb,8)
            traj{ii}(k,10) = 3;
        elseif vehtraj(ub,8)<vehtraj(ind,8) || vehtraj(ind,8)<vehtraj(lb,8)
            traj{ii}(k,10) = 2;
        else
            traj{ii}(k,10) = 1;
        end
        
        
       %% Get longitudinal maneuver:
        ub = min(size(vehtraj,1),ind+50);
        lb = max(1, ind-30);
        if ub==ind || lb ==ind
            traj{ii}(k,11) =1;
        else
            vHist = (vehtraj(ind,5)-vehtraj(lb,5))/(ind-lb);
            vFut = (vehtraj(ub,5)-vehtraj(ind,5))/(ub-ind);
            if vFut/vHist <0.8
                traj{ii}(k,11) =2;
            elseif vFut/vHist > 1.25
                traj{ii}(k,11) = 3;
            else
                traj{ii}(k,11) =1;
            end
        end
        % Get 
        % Get grid locations:
        t = vehTimes{ii}(int2str(time));
        frameEgo = t(t(:,8) == lane,:);
        frameL = t(t(:,8) == lane-1,:);
        frameR = t(t(:,8) == lane+1,:);
        if ~isempty(frameL)
            for l = 1:size(frameL,1)
                y = frameL(l,5)-traj{ii}(k,5);
                if abs(y) <90
                    gridInd = 1+round((y+90)/15);
                    traj{ii}(k,11+gridInd) = frameL(l,2);
                end
            end
        end
        for l = 1:size(frameEgo,1)
            y = frameEgo(l,5)-traj{ii}(k,5);
            if abs(y) <90 && y~=0
                gridInd = 14+round((y+90)/15);
                traj{ii}(k,11+gridInd) = frameEgo(l,2);
            end
        end
        if ~isempty(frameR)
            for l = 1:size(frameR,1)
                y = frameR(l,5)-traj{ii}(k,5);
                if abs(y) <90
                    gridInd = 27+round((y+90)/15);
                    traj{ii}(k,11+gridInd) = frameR(l,2);
                end
            end
        end
        
    end
end

save('allData_s','traj');


%% Split train, validation, test
load('./dataset/allData','traj');
disp('Splitting into train, validation and test sets...')

tracks = {};
trajAll = [];
for k = 1:6
    vehIds = unique(traj{k}(:, 2));
    for l = 1:length(vehIds)
        vehTrack = traj{k}(traj{k}(:, 2)==vehIds(l), :);
        tracks{k,vehIds(l)} = vehTrack(:, 3:11)'; % features and maneuver class id
        filtered = vehTrack(30+1:end-50, :);
        trajAll = [trajAll; filtered];
    end
end
clear traj;

trajTr=[];
trajVal=[];
trajTs=[];
for ii = 1:6
    no = trajAll(find(trajAll(:,1)==ii),:);
    len1 = length(no)*0.8;
    len2 = length(no)*0.9;
    trajTr = [trajTr;no(1:len1,:)];
    trajVal = [trajVal;no(len1:len2,:)];
    trajTs = [trajTs;no(len2:end,:)];
end

disp('Saving mat files...')
%%
traj = trajTr;
save('TrainSet','traj','tracks');

traj = trajVal;
save('ValSet','traj','tracks');

traj = trajTs;
save('TestSet','traj','tracks');










