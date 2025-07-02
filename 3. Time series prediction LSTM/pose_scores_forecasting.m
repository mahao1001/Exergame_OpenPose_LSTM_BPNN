close all; clear;clc
%% 加载模型
load('LSTM_Model_Trained_v7.mat');  % 包含 net
data = xlsread('pose_accuracy_scores_process.xlsx');
data = data(5:end, 2);

%% 参数
kim = 15;
zim = 1;
N = 35;

% === 构建训练用 res 矩阵，用于归一化拟合 ===
res = [];
for i = 1:(length(data) - kim - zim + 1)
    res(i, :) = [reshape(data(i:i+kim-1), 1, kim), data(i+kim+zim-1)];
end

X_train_all = res(:, 1:kim)';  % 每列是一个样本
[T_train_all, ps_output] = mapminmax(res(:, end)', 0, 1);  % 输出归一化
[X_train_norm, ps_input] = mapminmax(X_train_all, 0, 1);   % 输入归一化

%% 构建最后窗口作为预测起点
input_seq = data(end - kim + 1:end);             % 15×1
input_seq = reshape(input_seq, [1, kim]);        % 1×15
input_norm = mapminmax('apply', input_seq', ps_input);  % 15×1，不再 reshape

future_scores_norm = zeros(N, 1);
for i = 1:N
    input_cell = {input_norm};                           % {15×1}
    pred_norm = predict(net, input_cell);                % 预测
    future_scores_norm(i) = pred_norm;

    % 滑动窗口更新
    input_norm = [input_norm(2:end); pred_norm];
end

% === 反归一化
future_scores = mapminmax('reverse', future_scores_norm', ps_output);  % 1×N

%% 拼接序列用于绘图
full_sequence = [data; future_scores'];
x = 1:length(full_sequence);


%% === 计算未来预测平均分
avg_future_score = mean(future_scores);

% 取预测中间位置作为标注点
mid_point = length(data) + round(N / 2);
%% 可视化完整预测
figure;
plot(1:length(data), data, 'b-o', 'LineWidth', 1); hold on;
plot(length(data)+1:length(full_sequence), future_scores, 'r-*', 'LineWidth', 2);
plot(x, full_sequence, 'k--');
plot(mid_point - 10, avg_future_score, 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'y'); % 在图上标出预测平均值
text(mid_point - 6, avg_future_score, ...
     [ 'Score: ',  num2str(avg_future_score, '%.2f')], ...
     'FontSize', 12, 'Color', 'k');
xline(length(data), '--k', 'Prediction starting point', 'LabelVerticalAlignment', 'bottom');
legend('Historical scores', 'Future prediction score', 'Overall future trend','Future mean score','Location','northeast');
xlabel('Frame index'); ylabel('Posture score');
% title(['Full Pose Scoring Sequence + Future ', num2str(N), ' Frame Prediction（LSTM）']);
% axis tight;
xlim([0,410]);
ylim([40,100]);
grid on;

