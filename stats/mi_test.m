% max_lag = 100; % Maximum lag to consider
% mi_values = zeros(1, max_lag); % Preallocate mutual information array
% 
% for tau = 1:max_lag
%     x1 = x(1:end-tau);
%     x2 = x(tau+1:end);
%     mi_values(tau) = mi(x1, x2, 64); % Using 64 bins for histogram
% end
% 
% [~, optimal_tau] = min(mi_values); % Find the time delay with the first minimum mutual information
% disp(['Optimal time delay (tau): ', num2str(optimal_tau)]);

max_lag = 100; % Maximum lag to consider

% Autocorrelation
[acf, lags] = autocorr(x, 'NumLags', max_lag);
figure;
plot(lags, acf, '-o');
xlabel('Lag');
ylabel('Autocorrelation');
title('Autocorrelation Function');

% Mutual Information
mi_values = zeros(1, max_lag);
for tau = 1:max_lag
    x1 = x(1:end-tau);
    x2 = x(tau+1:end);
    mi_values(tau) = mi(x1, x2, 64); % Using 64 bins for histogram
end
figure;
plot(1:max_lag, mi_values, '-o');
xlabel('Lag');
ylabel('Mutual Information');
title('Mutual Information Function');