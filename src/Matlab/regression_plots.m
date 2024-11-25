clc; clear;

% Load the data
data = readtable("E:\PhD Work (Local)\Sit to Stand Fall Risk\sit2stand-fall\stats\dataClean.csv");

% Extract Age and input parameters
age = data.Age;
input_params = data(:, {'left_knee_max_mean_stand2sit', 'neck_min_y_acc', ...
                        'pelvic_avg_speed', 'trunk_lean_min_ang_acc', ...
                        'right_ankle_max_sd_stand2sit', 'left_knee_min_ang_vel_stand2sit', ...
                        'falling_1', 'time_stand2sit', 'Meditation', ...
                        'left_hip_min_ang_vel_sit2stand', 'right_ankle_max_mean_stand2sit', ...
                        'falling_2', 'pelvic_avg_y_speed', ...
                        'left_shank_angle_min_sd_stand2sit', 'right_ankle_ang_acc', ...
                        'pelvic_avg_y_acc', 'vigMins', 'right_hip_max', ...
                        'right_hip_min_ang_acc', 'numMedCond', 'left_shank_angle_ang_acc', ...
                        'neck_avg_y_acc', 'speed_sd', 'walkTime', 'IPAQ_SF_5', ...
                        'pelvic_min_speed_sit2stand'});

% Get the number of parameters
param_names = input_params.Properties.VariableNames;
num_params = width(input_params);

% Loop through each parameter and perform linear regression
for i = 1:num_params
    % Extract current parameter
    param = input_params{:, i};
    
    % Fit linear model
    lm = fitlm(age, param);
    
    % Remove outliers based on the model
    outliers = lm.Diagnostics.CooksDistance > 4/length(age);  % Identify outliers
    age_clean = age(~outliers);  % Remove outliers from age
    param_clean = param(~outliers);  % Remove outliers from parameter

    % Fit the model again after removing outliers
    lm_clean = fitlm(age_clean, param_clean);
    
    % Calculate Pearson correlation coefficient (R) and p-value
    [R, pValue] = corr(age_clean, param_clean);

    % Plot the results
    figure;
    plot(lm_clean);
    title(sprintf('Regression of %s vs Age', param_names{i}), 'Interpreter', 'none');
    xlabel('Age');
    ylabel(param_names{i});
    
    % Display R and p-value on the plot
    text(0.1 * max(age_clean), 0.9 * max(param_clean), sprintf('R = %.2f, p = %.3f', R, pValue), 'FontSize', 12);

    % Display regression statistics
    disp(lm_clean);
    
    pause(1); % Pause to view each plot before continuing
end
