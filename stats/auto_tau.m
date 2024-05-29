% MATLAB code to find time delay using autocorrelation
[acf, lags] = autocorr(x, 'NumLags', 100); % Compute autocorrelation
tau = find(acf < 1/exp(1), 1); % Find the first lag where acf falls below 1/e
disp(['Optimal time delay (tau): ', num2str(tau)]);