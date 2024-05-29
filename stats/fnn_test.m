tau = 9; % Use the tau from mutual information
max_m = 20; % Maximum embedding dimension to consider
Rtol = 15; % Radius tolerance for FNN criterion
Atol = 2; % Absolute tolerance for FNN criterion

% Compute FNN percentages
fnn_percent = fnn(x, max_m, tau, Rtol, Atol);

% Find the optimal embedding dimension
optimal_m = find(fnn_percent < 0.01, 1); % Threshold set to 1%

% Plot the FNN percentages
figure;
plot(1:max_m, fnn_percent, '-o');
xlabel('Embedding Dimension (m)');
ylabel('Percentage of False Nearest Neighbors');
title('False Nearest Neighbors Method');
grid on;

disp(['Optimal embedding dimension (m): ', num2str(optimal_m)]);
