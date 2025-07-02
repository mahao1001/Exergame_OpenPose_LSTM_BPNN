clc
clear all
rng(0);    % 固定随机种子确保可重复性

%% 1.示例数据（含缺失值、异常值和噪声）
raw_power = xlsread('待处理原始数据.xlsx'); % （含缺失值、异常值和噪声）的待处理原始数据
raw_power = raw_power(randperm(length(raw_power)));
t = 1:numel(raw_power);
% % 添加缺失值
% raw_power(randperm(200, 50)) = NaN;
% 
% % 添加异常值
% raw_power(randperm(200, 20)) = raw_power(randperm(100, 20)) + 12*rand(1,20);

%% 2.预处理步骤
% 步骤1：缺失值填补（线性插值）
filled_power = fillmissing(raw_power, 'linear');

% 步骤2：异常值处理（Hampel滤波器）
window_size = 2; % 滑动窗口大小
[cleaned_power, outlier_indices] = hampel(filled_power, window_size);

% 步骤3：数据平滑（移动平均）
smoothed_power = smoothdata(cleaned_power, 'movmean', window_size);

% 步骤4：数据标准化（Z-Score）
normalized_power = zscore(smoothed_power);

% 步骤5：最终数据对齐（时间轴修正）
final_power = normalized_power(window_size:end-window_size);
t_final = t(window_size:end-window_size);

%% 3.可视化对比
figure('Color','w','Position', [100,10,1200,800])

% 原始数据 vs 缺失值处理
subplot(6,1,1)
plot(t, raw_power, 'Color',[0.5 0.5 0.5],'LineWidth',2)
legend('Original data','Location','northwest','FontSize',12)
subplot(6,1,2)
plot(t, filled_power, 'b','LineWidth',2),
legend('After linear interpolation','Location','northwest','FontSize',12)
title('Step 1: Missing value handling','FontSize',12)

box off

% 缺失值处理后 vs 异常值处理
subplot(6,1,3)
plot(t, filled_power, 'b','LineWidth',2)
hold on
plot(t, cleaned_power, 'r','LineWidth',2)
scatter(t(outlier_indices), cleaned_power(outlier_indices), 40, 'm', 'filled')
title('Step 2: Outlier processing (Hampel filtering)','FontSize',12)
legend('Before processing', 'After processing', 'Correction point','Location','northwest','FontSize',10)
box off

% 异常值处理后 vs 数据平滑
subplot(6,1,4)
plot(t, cleaned_power, 'r','LineWidth',2)
hold on
plot(t, smoothed_power, 'Color',[0 0.6 0],'LineWidth',2)
title('Step 3: Data smoothing (moving average)','FontSize',12)
legend('Original data', 'Smoothed data','Location','northwest','FontSize',12)
box off

% 数据平滑 vs 标准化
subplot(6,1,5)
yyaxis left
plot(t, smoothed_power, 'Color',[0 0.6 0],'LineWidth',2)
ylabel('Original Range','FontSize',12)
yyaxis right
plot(t, normalized_power, 'm','LineWidth',2)
ylabel('Normalized value','FontSize',12)
title('Step 4: Data Standardization (Z-Score)','FontSize',12)
legend('Smoothed data', 'Standardized data','Location','northwest','FontSize',12)
box off

% 最终效果对比
subplot(6,1,6)
plot(t, raw_power, 'Color',[0.5 0.5 0.5],'LineWidth',2)
hold on
plot(t_final, final_power, 'r','LineWidth',2)
title('Step 5: Final comparison','FontSize',12)
legend('Original data', 'Preprocessed data','Location','northwest','FontSize',12)
box off
