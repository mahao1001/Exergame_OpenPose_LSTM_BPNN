close all;clear;clc

%% 读取 JSON 文件
jsonText = fileread('20250618boxingCOM.poses.json');
jsonData = jsondecode(jsonText);

model_all = [];
athlete_all = [];

%% 提取每帧关键点（标准动作在左，模拟动作在右）
for i = 1:length(jsonData)
    frame = jsonData(i);
    if length(frame.subset) >= 2
        subset = frame.subset;
        candidate = frame.candidate;

        model_idx = subset(2, 1:18) + 1;     % 左边模特
        athlete_idx = subset(1, 1:18) + 1;   % 右边运动员

        model_pts = nan(18, 2);
        athlete_pts = nan(18, 2);

        for k = 1:18
            if model_idx(k) > 0 && model_idx(k) <= size(candidate, 1)
                model_pts(k, :) = candidate(model_idx(k), 1:2);
            end
            if athlete_idx(k) > 0 && athlete_idx(k) <= size(candidate, 1)
                athlete_pts(k, :) = candidate(athlete_idx(k), 1:2);
            end
        end

        model_all(:, :, end+1) = model_pts';
        athlete_all(:, :, end+1) = athlete_pts';
    end
end

%% 姿态归一化（用颈部和髋部估算身高后除以）
model_norm = model_all;
athlete_norm = athlete_all;
nFrames = size(model_all, 3);

for i = 1:nFrames
    % 模特归一化
    p = model_all(:, :, i);
    if all(~isnan(p(:,2))) && all(~isnan(p(:,9)))
        scale = norm(p(:,2) - p(:,9));
        if scale > 1e-3
            model_norm(:,:,i) = p / scale;
        end
    end

    % 运动员归一化
    p2 = athlete_all(:, :, i);
    if all(~isnan(p2(:,2))) && all(~isnan(p2(:,9)))
        scale2 = norm(p2(:,2) - p2(:,9));
        if scale2 > 1e-3
            athlete_norm(:,:,i) = p2 / scale2;
        end
    end
end

%% 计算每帧的平均欧氏距离误差
err = zeros(1, nFrames);
for i = 1:nFrames
    delta = athlete_norm(:,:,i) - model_norm(:,:,i);
    d = sqrt(sum(delta.^2, 1));  % 每个关键点的欧氏距离
    d(isnan(d)) = [];            % 删除NaN，避免影响平均值
    if ~isempty(d)
        err(i) = mean(d);
    else
        err(i) = NaN;            % 若全是NaN则保留空值
    end
end

%% 归一化为准确率评分
%% 使用模特身高（点2到点9）归一化误差为相对误差
bone_pairs = [
     1 2;       % Nose - Neck
    2 3; 3 4;  % Neck - RShoulder - RElbow
    4 5;       % RElbow - RWrist
    2 6; 6 7;  % Neck - LShoulder - LElbow
    7 8;       % LElbow - LWrist
    2 9;       % Neck - MidHip
    9 10; 10 11;  % MidHip - RHip - RKnee
    11 12;        % RKnee - RAnkle
    9 13; 13 14;  % MidHip - LHip - LKnee
    14 15;        % LKnee - LAnkle
    1 16; 1 17;   % Nose - REye, Nose - LEye
    16 18  % REye - Ear, LEye - Ear
];

skeleton_lengths = zeros(1, nFrames);
for i = 1:nFrames
    p = model_norm(:,:,i);  % 使用未归一化的模特数据
    len = 0;
    for j = 1:size(bone_pairs,1)
        a = bone_pairs(j,1);
        b = bone_pairs(j,2);
        if all(~isnan(p(:,a))) && all(~isnan(p(:,b)))
            len = len + norm(p(:,a) - p(:,b));
        end
    end
    skeleton_lengths(i) = len;
end
skeleton_length = max(skeleton_lengths);

% 转为准确率评分（1代表动作一致，0代表完全不同）
normalized_err = err ./ skeleton_lengths;
accuracy = 1 - normalized_err;
accuracy = accuracy+0.15;
accuracy(accuracy < 0) = 0;  % 避免负值
% accuracy_smooth = movmean(accuracy, 5);  % 5帧滑动窗口平均
% accuracy_smooth = medfilt1(accuracy, 5);  % 5点中值滤波
accuracy_smooth = movmean(medfilt1(accuracy, 3), 5);  %双重滤波（更平稳）
%% 绘图显示准确率曲线
figure;
plot(accuracy, ':b', 'DisplayName', 'Accuracy', 'LineWidth', 2); hold on;
plot(accuracy_smooth, 'r', 'DisplayName', 'Smoothed Accuracy', 'LineWidth', 2);
xlabel('Frame Index'); ylabel('Pose Accuracy');
title('Simulated vs. Standard Pose Accuracy');
legend show;
axis tight;
ylim([0,1]);
grid on;

%% 可选：导出准确率为CSV
T = table((1:nFrames)', accuracy_smooth', 'VariableNames', {'Frame', 'Accuracy'});
writetable(T, 'pose_accuracy_scores.csv');
