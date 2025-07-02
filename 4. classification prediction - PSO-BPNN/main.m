%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = xlsread('boxing_Wrist.xlsx');  % 8输入1输出


%%  划分训练集和测试集
temp = randperm(size(res,1));

P_train = res(temp(1: floor(size(res,1)*0.7)), 1: size(res,2)-1)';
T_train = res(temp(1: floor(size(res,1)*0.7)), size(res,2))';
M = size(P_train, 2);

P_test = res(temp(floor(size(res,1)*0.7)+1: end), 1: size(res,2)-1)';
T_test = res(temp(floor(size(res,1)*0.7)+1: end), size(res,2))';
N = size(P_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test  = mapminmax('apply', P_test, ps_input);
t_train = ind2vec(T_train);
t_test  = ind2vec(T_test );

%%  节点个数
inputnum  = size(p_train, 1);  % 输入层节点数
hiddennum = 10;                 % 隐藏层节点数
outputnum = size(t_train, 1);  % 输出层节点数

%%  建立网络
net = newff(p_train, t_train, hiddennum);

%%  设置训练参数
net.trainParam.epochs     = 1000;      % 训练次数
net.trainParam.goal       = 1e-6;      % 目标误差
net.trainParam.lr         = 0.01;      % 学习率
net.trainParam.showWindow = 0;         % 关闭窗口

%%  参数初始化
c1      = 4.494;       % 学习因子
c2      = 4.494;       % 学习因子
maxgen  =   30;        % 种群更新次数  
sizepop =    5;        % 种群规模
Vmax    =  1.0;        % 最大速度
Vmin    = -1.0;        % 最小速度
popmax  =  2.0;        % 最大边界
popmin  = -2.0;        % 最小边界

%%  节点总数
numsum = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum;

for i = 1 : sizepop
    pop(i, :) = rands(1, numsum);  % 初始化种群
    V(i, :) = rands(1, numsum);    % 初始化速度
    fitness(i) = fun(pop(i, :), hiddennum, net, p_train, t_train);
end

%%  个体极值和群体极值
[fitnesszbest, bestindex] = min(fitness);
zbest = pop(bestindex, :);     % 全局最佳
gbest = pop;                   % 个体最佳
fitnessgbest = fitness;        % 个体最佳适应度值
BestFit = fitnesszbest;        % 全局最佳适应度值

%%  迭代寻优
for i = 1 : maxgen
    for j = 1 : sizepop
        
        % 速度更新
        V(j, :) = V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
        V(j, (V(j, :) > Vmax)) = Vmax;
        V(j, (V(j, :) < Vmin)) = Vmin;
        
        % 种群更新
        pop(j, :) = pop(j, :) + 0.2 * V(j, :);
        pop(j, (pop(j, :) > popmax)) = popmax;
        pop(j, (pop(j, :) < popmin)) = popmin;
        
        % 自适应变异
        pos = unidrnd(numsum);
        if rand > 0.95
            pop(j, pos) = rands(1, 1);
        end
        
        % 适应度值
        fitness(j) = fun(pop(j, :), hiddennum, net, p_train, t_train);

    end
    
    for j = 1 : sizepop

        % 个体最优更新
        if fitness(j) < fitnessgbest(j)
            gbest(j, :) = pop(j, :);
            fitnessgbest(j) = fitness(j);
        end

        % 群体最优更新 
        if fitness(j) < fitnesszbest
            zbest = pop(j, :);
            fitnesszbest = fitness(j);
        end

    end

    BestFit = [BestFit, fitnesszbest];    
end

%%  提取最优初始权值和阈值
w1 = zbest(1 : inputnum * hiddennum);
B1 = zbest(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);
w2 = zbest(inputnum * hiddennum + hiddennum + 1 : inputnum * hiddennum ...
    + hiddennum + hiddennum * outputnum);
B2 = zbest(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
    inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);

%%  网络赋值
net.Iw{1, 1} = reshape(w1, hiddennum, inputnum );
net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);
net.b{1}     = reshape(B1, hiddennum, 1);
net.b{2}     = B2';

%%  打开训练窗口 
net.trainParam.showWindow = 1;        % 打开窗口

%%  网络训练
net = train(net, p_train, t_train);

%%  仿真预测
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test );

%%  数据反归一化
T_sim1 = vec2ind(t_sim1);
T_sim2 = vec2ind(t_sim2);

%%  数据排序
[T_train, index_1] = sort(T_train);
[T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

%%  性能评价
error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('Actual value', 'Predicted value')
xlabel('Samples of training set')
ylabel('Prediction results')
string = {'Comparison of prediction results of training set'; ['Accuracy rate=' num2str(error1) '%']};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('Actual value', 'Predicted value')
xlabel('Samples of test set')
ylabel('Prediction results')
string = {'Comparison of prediction results of test set'; ['Accuracy rate=' num2str(error2) '%']};
title(string)
xlim([1, N])
grid

%%  误差曲线迭代图
figure
plot(1: length(BestFit), BestFit, 'LineWidth', 1.5);
xlabel('The number of iterations');
ylabel('Fitness value');
xlim([1, length(BestFit)])
string = {'Model iteration error'};
title(string)
grid on

%%  混淆矩阵
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
    
figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';