%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据（时间序列的单列数据）
result = xlsread('pose_accuracy_scores_process.xlsx');
result = result(:, 2);
%%  数据分析
num_samples = length(result);  % 样本个数 
kim = 15;                      % 延时步长（kim个历史数据作为自变量）
zim =  1;                      % 跨zim个时间点进行预测

%%  划分数据集
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(result(i: i + kim - 1), 1, kim), result(i + kim + zim - 1)];
end

%%  划分训练集和测试集
num_total = size(res, 1);
train_ratio = 0.7;  % 70%训练集
train_count = floor(num_total * train_ratio);

P_train = res(1:train_count, 1:15)';
T_train = res(1:train_count, 16)';
M = size(P_train, 2);

P_test  = res(train_count+1:end, 1:15)';
T_test  = res(train_count+1:end, 16)';
N = size(P_test, 2);

%%  数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  数据平铺
% 将数据平铺成1维数据只是一种处理方式
% 也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
% 但是应该始终和输入层数据结构保持一致
P_train =  double(reshape(P_train, 15, 1, 1, M));
P_test  =  double(reshape(P_test , 15, 1, 1, N));

t_train = t_train';
t_test  = t_test' ;

%%  数据格式转换
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end

%%  创建模型
layers = [
    sequenceInputLayer(15)              % 建立输入层
    
    lstmLayer(10, 'OutputMode', 'last') % LSTM层
    reluLayer                           % Relu激活层
    
    fullyConnectedLayer(1)              % 全连接层
    regressionLayer];                   % 回归层
 
%%  参数设置
options = trainingOptions('adam', ...       % Adam 梯度下降算法
    'MaxEpochs', 1000, ...                  % 最大训练次数
    'InitialLearnRate', 5e-3, ...           % 初始学习率
    'LearnRateSchedule', 'piecewise', ...   % 学习率下降
    'LearnRateDropFactor', 0.1, ...         % 学习率下降因子
    'LearnRateDropPeriod', 800, ...         % 经过 800 次训练后 学习率为 0.005 * 0.1
    'Shuffle', 'every-epoch', ...           % 每次训练打乱数据集
    'Plots', 'training-progress', ...       % 画出曲线
    'Verbose', false);

%%  训练模型
net = trainNetwork(p_train, t_train, layers, options);

%%  仿真预测
t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test );

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  均方根误差
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);

%%  查看网络结构
analyzeNetwork(net)

%%  绘图
figure
plot(1: M, T_train, 'r-', 1: M, T_sim1, 'b-', 'LineWidth', 1)
legend('Actual value', 'Predicted value')
xlabel('Frame index')
ylabel('Predicted score results')
string = {'Comparison of training set prediction results'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
ylim([60,95])
grid

figure
plot(1: N, T_test, 'r-', 1: N, T_sim2, 'b-', 'LineWidth', 1)
legend('Actual value', 'Predicted value')
xlabel('Frame index')
ylabel('Predicted score results')
string = {'Comparison of prediction results of test set'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
ylim([60,95])
grid

%%  相关指标计算
% R2
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;

disp(['The R2 of the training set data is：', num2str(R1)])
disp(['The R2 of the test set data is：', num2str(R2)])

% MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M ;
mae2 = sum(abs(T_sim2' - T_test )) ./ N ;

disp(['The MAE of the training set data is：', num2str(mae1)])
disp(['The MAE of the test set data is：', num2str(mae2)])

% MBE
mbe1 = sum(T_sim1' - T_train) ./ M ;
mbe2 = sum(T_sim2' - T_test ) ./ N ;

disp(['The MBE of the training set data is：', num2str(mbe1)])
disp(['The MBE of the test set data is：', num2str(mbe2)])

%%  绘制散点图
sz = 25;
c = 'b';

figure
scatter(T_train, T_sim1, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('The actual value of the training set');
ylabel('The predicted value of the training set');
xlim([min(T_train) max(T_train)])
ylim([min(T_sim1) max(T_sim1)])
title('Training set predicted values ​​vs. training set actual values')

figure
scatter(T_test, T_sim2, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('The actual value of the test set');
ylabel('The predicted value of the test set');
xlim([min(T_test) max(T_test)])
ylim([min(T_sim2) max(T_sim2)])
title('Test set predicted value vs. test set true value')

save('LSTM_Model_Trained_v7.mat', 'net', 'ps_input', 'ps_output', 'kim', 'result', '-v7.3');
