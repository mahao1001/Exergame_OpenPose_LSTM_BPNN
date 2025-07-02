clear; clc; close all;

%% 1. 参数设置
jsonFile = '20250618boxingCOM.poses.json';   % ← 你的 JSON 文件
if ~isfile(jsonFile)
    error("找不到 JSON 文件：%s", jsonFile);
end


%% 2. 读取并解析 JSON
txt = fileread(jsonFile);
raw = jsondecode(txt);
if iscell(raw)
    data = [raw{:}];
elseif isstruct(raw)
    data = raw;
else
    error("JSON 最外层既不是 list 也不是 struct，请检查格式。");
end
F = numel(data);
fprintf("载入 JSON，检测到 %d 帧数据。\n", F);

%% 3. 整帧（整体）置信度提取
personScore = nan(F,1);
for i = 1:F
    sub = data(i).subset;
    if iscell(sub), sub = cell2mat(sub); end
    if ~isempty(sub)
        % 取第一个人的整帧置信度（倒数第二列）
        personScore(i) = sub(1,end-1);
    end
end
% --- Min–Max 归一化到 [0,1] ---
minS = min(personScore);
maxS = max(personScore);
normPerson = (personScore - minS) / (maxS - minS);

% 平滑
w = 5;  % 滑动窗口大小
smoothPerson = movmean(personScore, w);
% 同样对平滑曲线归一化
normSmooth = (smoothPerson - minS) / (maxS - minS);

%% 4. 单关节点置信度提取（示例：Nose=1, Neck=2, R_Shoulder=3）
jointNames   = {'Nose','R-Shoulder','L-Shoulder'};
jointCOCOidx = [1,3,6];  % COCO 中的 0-based idx 再 +1 成 MATLAB idx
J = numel(jointCOCOidx);
jointScore = nan(F, J);

for i = 1:F
    cand = data(i).candidate;
    if iscell(cand), cand = cell2mat(cand); end
    sub  = data(i).subset;
    if iscell(sub),  sub  = cell2mat(sub);  end

    if ~isempty(sub) && ~isempty(cand)
        idxs0 = sub(1, 1:18);  % 取第一个人前18个关节点的 0-based 索引
        for j = 1:J
            ci = idxs0(jointCOCOidx(j));
            if ci >= 0
                jointScore(i,j) = cand(ci+1, 3);  % +1 修正到 MATLAB 索引，3 列为 score
            end
        end
    end
end

%% 5. 绘图
t = (1:F)';

figure('Name','Confidence Curves','NumberTitle','off','Position',[100 100 800 600]);

% 整帧置信度
ax1 = subplot(2,1,1);
% plot(ax1, t, personScore, '-b', 'DisplayName','Raw Person Score'); hold(ax1,'on');
% plot(ax1, t, smoothPerson, '-r', 'LineWidth',2, 'DisplayName',sprintf('%d-frame MA',w));
plot(t, normPerson, '-b','DisplayName','Normalized Score'); hold on;
plot(t, normSmooth, '-r','LineWidth',2,'DisplayName',sprintf('%d-frame MA',w));
xlabel(ax1,'Frame'); ylabel(ax1,'Person Confidence');
title(ax1,'Posture detection confidence time series');
legend(ax1,'Location','southeast'); 
grid(ax1,'on');
axis tight;
ylim([0 1]);



% 单关节点置信度
ax2 = subplot(2,1,2);
colors = [ 0 0.4470 0.7410             % 蓝色  
           0.4660, 0.6740, 0.1880      % 绿色
           0.8500 0.3250 0.0980       % 红色
                ];  
hold(ax2,'on');
for j = 1:J
    plot(ax2, t, jointScore(:,j), '-', 'Color',colors(j,:), 'DisplayName',jointNames{j},'LineWidth',1.2);
end
xlabel(ax2,'Frame'); ylabel(ax2,'Keypoint Confidence');
title(ax2,'Single joint confidence curve');
legend(ax2,'Location','southeast'); 
grid(ax2,'on');
axis tight;
ylim([0.7 1]);
hold off;

% 整体标题
% sgtitle('姿态检测置信度时序');

% 自动调整子图间距
ax1.Position(2) = ax1.Position(2) ;
ax2.Position(4) = ax2.Position(4) ;