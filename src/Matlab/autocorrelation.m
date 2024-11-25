clc; clear;

data_path = "E:\PhD Work (Local)\Sit to Stand Fall Risk\data\matlab_data\cleaned";
mat_files = dir(fullfile(data_path, '*.mat'));
%file_path = fullfile(data_path, mat_files(10).name);
file_path = fullfile(data_path, 'm_wBZ1RW0f.mat');
file = load(file_path);

data = double(file.data); % [frames , joints , (x,
% y)]
framerate = double(file.framerate);
data_norm = sqrt(sum(data.^2, 3)); % [frames , joints]
j = 10;

signal = data_norm(:, j) - mean(data_norm(:, j));
[Acr, lags] = xcov(signal, signal);
Acr = Acr / max(Acr);
[pks, locs] = findpeaks(Acr);
pks2 = pks(pks > 0);
locs2 = locs(pks > 0);
arr = lags(locs);
index_of_zero = find(arr == 0);

half_cycle_regularity = pks2(index_of_zero + 1);
full_cycle_regularity = pks2(index_of_zero + 2);
symmetry_1 =  half_cycle_regularity/full_cycle_regularity;
symmetry_2 = abs(half_cycle_regularity - full_cycle_regularity);

subplot(2,1,1); 
plot(data_norm(:, j));
xlim([0, lags(end)])
xlabel('Time');
ylabel('Right Hip Joint');

subplot(2,1,2);
hold on;
plot(lags, Acr);
plot([lags(1), lags(end)], [0, 0])
plot(lags(locs2), pks2, 'ro');
hold off;
xlim([lags(1), lags(end)])
xlabel('Time');
ylabel('Autocorrelation');