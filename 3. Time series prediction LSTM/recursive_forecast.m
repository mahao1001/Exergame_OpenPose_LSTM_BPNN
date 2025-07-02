function future_scores = recursive_forecast(net, data, ps_output, kim, N)
    % net: 训练好的 LSTM 网络
    % data: 原始评分序列（未归一化）
    % ps_output: 输出归一化结构体
    % kim: 输入步长（窗口长度）
    % N: 预测未来帧数

    % 输入归一化参数
    [~, ps_input] = mapminmax(data', 0, 1);  % 输入统一归一化

    % 初始窗口归一化
    input_window = data(end - kim + 1:end);  
    input_window_norm = mapminmax('apply', input_window', ps_input);  % [1×15]

    % 初始化预测结果
    future_scores_norm = zeros(N, 1);

    for i = 1:N
        input_seq = reshape(input_window_norm, [15, 1]);  % [15×1]
        input_cell = {input_seq};

        pred_norm = predict(net, input_cell);             % LSTM预测
        future_scores_norm(i) = pred_norm;

        % 更新滑动窗口
        input_window_norm = [input_window_norm(2:end), pred_norm];
    end

    % 反归一化
    future_scores = mapminmax('reverse', future_scores_norm', ps_output);  % [1×N]

    % 绘图
    figure;
    plot(length(data)-kim+1:length(data), data(end-kim+1:end), 'b', 'LineWidth', 2); hold on;
    plot(length(data)+1:length(data)+N, future_scores, 'r--o', 'LineWidth', 2);
    legend('历史评分序列', '预测评分');
    xlabel('时间帧'); ylabel('姿态评分');
    title(['基于 LSTM 的未来 ', num2str(N), ' 帧预测']);
    grid on;
end
