clear; clc

data_path = "E:\PhD Work (Local)\Sit to Stand Fall Risk\data\matlab_data\cleaned";

mat_files = dir(fullfile(data_path, '*.mat'));

% Preallocation
numFiles = length(mat_files);
results(numFiles).id = 0;
results(numFiles).fileName = '';  
results(numFiles).lyapunovExponent = 0;
results(numFiles).approxEntropy = 0;
results(numFiles).sampleEntropy = 0;

for k = 1:length(mat_files)
    % Get the full path to the .mat file
    file_path = fullfile(data_path, mat_files(k).name);

    % Remove the '.mat' extension to get the file name
    [~, var_name, ~] = fileparts(mat_files(k).name);
    
    try
    % Load the .mat file into a structure
    file = load(file_path);

    % Convert to double
    data = double(file.data); % [frames , joints , (x,y)]
    framerate = double(file.framerate);
    s = size(data); % Size of data

    % Remove Left and Right Eyes and Ears Keypoints
    data(:, [16, 17, 18, 19], :) = [];

    % Remove Right Ankle as it is just 0 (used as reference point)
    data(:, 12, :) = [];

    % Velocity
    vel = gradient(data, 1);

    % Acceleration
    acc = gradient(vel, 1);

    % Acceleration RMS
    acc_norm = sqrt(sum(acc.^2, 3));

    % Duplicate data to give more steps
    data2 = [data; data; data];
    
    % Display the name of the file being processed
    disp(['Processing file: ', mat_files(k).name]);

    % Euclidean Norm of the joint position vector
    data_norm = sqrt(sum(data.^2, 3)); % [frames , joints]
    
    % figure(k)
    % %phaseSpaceReconstruction(acc_norm(:,10))
    % [~, lag, dim] = phaseSpaceReconstruction(data_norm(:,10));
    % lyapunovExponent(data_norm(:,10),framerate,lag,dim, 'ExpansionRange', [1*round(s(1)/10), 2*round(s(1)/10)])
    % saveas(gcf, sprintf('./png/Lyapunov Exponent 2/lyapExp_2_%d.png', k))
    % close(gcf)
    
    % right hip -> 10 (if we start from 1)
    j = 10;
    [~, lag, dim] = phaseSpaceReconstruction(data_norm(:,j));
    lyap_Exp = lyapunovExponent(data_norm(:,j),framerate,lag,dim, 'ExpansionRange', [0*round(s(1)/10), 1*round(s(1)/10)]);
    app_Entropy = ApEn(data_norm(:,j));
    sample_Entropy = SampEn(data_norm(:,j));


    % Refined Multiscale [Sample] Entropy
    Mobj = MSobject('SampEn', 'm', 4, 'r', 1.25);
    [MSx, Ci] = rMSEn(data_norm(:,j), Mobj, 'Scales', 20, 'F_Order', 3, 'F_Num', 0.6, 'RadNew', 4);

    % Regularity and Symmetry
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


    % Saving in results structure
    results(k).id = k;
    results(k).fileName = var_name; 
    results(k).lyapunovExponent = lyap_Exp;
    results(k).approxEntropy = app_Entropy(end);
    results(k).sampleEntropy = sample_Entropy(end);
    results(k).MSE1 = MSx(1);
    results(k).MSE2 = MSx(2);
    results(k).MSE3 = MSx(3);
    results(k).MSE4 = MSx(4);
    results(k).MSE5 = MSx(5);
    results(k).MSE6 = MSx(6);
    results(k).MSE7 = MSx(7);
    results(k).MSE8 = MSx(8);
    results(k).MSE9 = MSx(9);
    results(k).MSE10 = MSx(10);
    results(k).MSE11 = MSx(11);
    results(k).MSE12 = MSx(12);
    results(k).MSE13 = MSx(13);
    results(k).MSE14 = MSx(14);
    results(k).MSE15 = MSx(15);
    results(k).MSE16 = MSx(16);
    results(k).MSE17 = MSx(17);
    results(k).MSE18 = MSx(18);
    results(k).MSE19 = MSx(19);
    results(k).MSE20 = MSx(20);
    results(k).Ci = Ci;
    results(k).half_cycle_regularity = half_cycle_regularity;
    results(k).full_cycle_regularity = full_cycle_regularity;
    results(k).symmetry_1 = symmetry_1;
    results(k).symmetry_2 = symmetry_2;

    catch ME
        % If there is an error, display it
        fprintf('Error loading file: %s\n', mat_files(k).name);
        disp(ME.message);
    end

end


% Check for non-scalar or empty Lyapunov exponent values and handle them
for i = 1:numFiles
    if isempty(results(i).lyapunovExponent) || ~isscalar(results(i).lyapunovExponent)
        results(i).lyapunovExponent = NaN;  % Assign NaN to invalid entries
    end
end

% Extract Lyapunov exponent values from results
lyapunovValues = [results.lyapunovExponent]; 

% Sort the Lyapunov exponents in descending order, ignoring NaN values
[~, sortedIdx_L] = sort(lyapunovValues, 'descend', 'MissingPlacement', 'last');  % NaN values go to the end

% Rearrange the results structure based on the sorted indices
sortedResults_L = results(sortedIdx_L);

T = struct2table(results);

writetable(T, 'MSE.csv')


% % Extract Entropy values from results
% entropyValues = [results.approxEntropy]; 
% 
% % Sort the Entropy exponents in descending order, ignoring NaN values
% [~, sortedIdx_E] = sort(entropyValues, 'descend', 'MissingPlacement', 'last');  % NaN values go to the end
% 
% % Rearrange the results structure based on the sorted indices
% sortedResults_E = results(sortedIdx_E);


% hold on;
% plot(lyapunovValues_norm);
% plot(lyapunovValues_x);
% plot(lyapunovValues_y);
% legend('norm', 'x', 'y')
