
% ===============================
% Boxing Punch Intensity Analysis
% ===============================

% Load dataset
data = readtable('boxing_punch_dataset.csv');

% Compute acceleration and gyroscope magnitudes
AccMag = sqrt(data.Mean_Acc_X.^2 + data.Mean_Acc_Y.^2 + data.Mean_Acc_Z.^2);
GyrMag = sqrt(data.Mean_Gyr_X.^2 + data.Mean_Gyr_Y.^2 + data.Mean_Gyr_Z.^2);

% Add to table
data.Acc_Magnitude = AccMag;
data.Gyr_Magnitude = GyrMag;

% Estimate energy expenditure (simple model)
alpha = 0.6; beta = 0.4;
data.Estimated_Energy = alpha * AccMag + beta * GyrMag;

% Plot distributions
figure;
subplot(2,1,1);
histogram(data.Acc_Magnitude, 30);
title('Acceleration Magnitude Distribution');
xlabel('Acc Magnitude'); ylabel('Count');

subplot(2,1,2);
plot(data.Estimated_Energy);
title('Estimated Energy per Punch');
xlabel('Sample'); ylabel('Energy');

% Save processed data
writetable(data, 'processed_boxing_dataset.csv');


%% 训练



