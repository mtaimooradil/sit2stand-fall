clc;clear;close;
data_path = "E:\PhD Work (Local)\Sit to Stand Fall Risk\data\matlab_data\cleaned";
mat_files = dir(fullfile(data_path, '*.mat'));
%file_path = fullfile(data_path, mat_files(10).name);
file_path = fullfile(data_path, 'm_a6j4D1CL.mat');
file = load(file_path);

data = double(file.data); % [frames , joints , (x,y)]
framerate = double(file.framerate);
data_norm = sqrt(sum(data.^2, 3)); % [frames , joints]
j = 10;

d = data_norm(:, j);
m = mean(data_norm, 1);
less_than_mean_indices = find(d < m(j));
less_than_mean = d(less_than_mean_indices);
greater_than_mean_indices = find(d > m(j));
greater_than_mean = d(greater_than_mean_indices);

figure(1); 
plot([0 length(d)], [m(j) m(j)])
hold on;
% plot(data_norm(:,j))
plot(less_than_mean_indices, less_than_mean)
plot(greater_than_mean_indices, greater_than_mean)
hold off;
figure(2);
[l_pks, l_locs] = findpeaks(less_than_mean, 'MinPeakProminence',0.05);
plot(less_than_mean)
hold on;
plot(l_locs, l_pks, 'ro')
hold off;
figure(3);
[g_pks, g_locs] = findpeaks(-greater_than_mean, 'MinPeakProminence',0.05);
plot(greater_than_mean)
hold on;
plot(g_locs, -g_pks, 'ro')
hold off;
figure(4);
plot(less_than_mean)
hold on;
plot(l_locs, l_pks, 'ro')
plot(greater_than_mean)
plot(g_locs, -g_pks, 'ro')
hold off;

% points
% initial = [1 21];
% c1 = [22 107];
% c2 = [108 192];
% c3 = [193 281];
% c4 = [282 368];
% c5 = [369 406];
% final = [407 439];

initial = [1 56];
c1 = [57 110];
c2 = [111 172];
c3 = [173 235];
final = [236 270];

initial_cycle = d(initial(1):initial(2));
cycle_1 = d(c1(1):c1(2));
cycle_2 = d(c2(1):c2(2));
cycle_3 = d(c3(1):c3(2));
% cycle_4 = d(c4(1):c4(2));
% cycle_5 = d(c5(1):c5(2));
% final_cycle = d(c5(1):final(2));
final_cycle = d(final(1):final(2));

% full_cycle = {initial_cycle, cycle_1, cycle_2, cycle_3, cycle_4, final_cycle};

full_cycle = {initial_cycle, cycle_1, cycle_2, cycle_3, final_cycle};

figure;
plot(initial(1):initial(2), initial_cycle)
hold on;
plot(c1(1):c1(2),cycle_1)
plot(c2(1):c2(2),cycle_2)
plot(c3(1):c3(2),cycle_3)
%plot(c4(1):c4(2),cycle_4)
% plot(c5(1):c5(2),cycle_5)
% plot(c5(1):final(2),final_cycle)
plot(final(1):final(2),final_cycle)
hold off;

% Extract the first and last cycle (keep them as is)
first_cycle = full_cycle{1};
last_cycle = full_cycle{end};

% Extract the middle cycles (i.e., cycle_1 to cycle_5)
middle_cycles = full_cycle(2:end-1); % This is a cell array of the middle cycles

% Initialize an empty cell array to store shuffled cycles
shuffled_cycles = {};

% Shuffle the middle cycles 20 times
for i = 1:20
    shuffled_indices = randperm(length(middle_cycles)); % Get random shuffle of indices
    shuffled_cycles = [shuffled_cycles, middle_cycles(shuffled_indices)]; % Append shuffled cycles
end

% Concatenate first cycle, shuffled middle cycles, and last cycle
long_cycle_vector = [first_cycle; vertcat(shuffled_cycles{:}); last_cycle];

vel = gradient(data, 1);
acc = gradient(vel, 1);
vel_norm = sqrt(sum(vel.^2, 3));
acc_norm = sqrt(sum(acc.^2, 3));

% j = 10;
% [~, lag, dim] = phaseSpaceReconstruction(vel_norm(:,j));
% lyap_Exp = lyapunovExponent(data_norm(:,j),framerate,lag,dim, 'ExpansionRange', [0*round(s(1)/10), 2*round(s(1)/10)]);
% 
% figure(1)
% plot(data(:,10,2), vel(:,10,2))
% figure(2)
% plot(data(:,10,1), vel(:,10,1))
% figure(3)
% plot(data_norm(:,10), vel_norm(:,10))
