
% 1. 读入整个 JSON 文本
jsonFile = '20250618boxingCOM.poses.json';
txt = fileread(jsonFile);

% 2. 解析成 MATLAB 结构体数组
data = jsondecode(txt);
% data 是 1×F 的结构体数组，F = 总帧数

% 3. 查看结构体里有哪些字段
disp(fieldnames(data))
%   → 'frame'    (double)
%     'candidate' (cell 或 numeric)
%     'subset'    (cell 或 numeric)

% 4. 遍历每一帧，提取关键点
F = numel(data);
allCandidates = cell(F,1);
allSubsets    = cell(F,1);
for i = 1:F
    % 当前帧索引（可选）
    frameID = data(i).frame;
    
    % 关键点列表 Nx3： [x, y, score]
    % 有时 jsondecode 会把它当 cell，视情况转换
    cand = data(i).candidate;
    if iscell(cand), cand = cell2mat(cand); end
    
    % subset 列表 Mx20，每行是一张人的关键点在 candidate 里的索引
    sub = data(i).subset;
    if iscell(sub), sub = cell2mat(sub); end
    
    allCandidates{i} = cand;
    allSubsets{i}    = sub;
    
    % （示例）把第 i 帧的第一个人关键点画出来
    if ~isempty(sub)
      firstPerson = sub(1,1:18);        % 取前 18 个点
      coords = cand(firstPerson+1,1:2); % +1 因为 Python 保存的是 0-base
      scatter(coords(:,1), coords(:,2), 'filled');
      title(sprintf('Frame %d, Person 1', frameID));
      axis ij; axis equal; drawnow;
    end
end

% 5. 处理完毕，你就得到了 allCandidates, allSubsets，可以做后续分析