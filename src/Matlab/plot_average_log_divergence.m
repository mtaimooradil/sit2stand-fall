function plot_average_log_divergence(signal, max_lag)
    % Ensure the signal is a column vector
    signal = signal(:);
    
    % Check if max_lag is within a valid range
    if max_lag <= 0 || max_lag >= length(signal)
        error('max_lag must be between 1 and length of the signal - 1');
    end

    % Initialize ALD values for different lags
    ald_values = zeros(max_lag, 1);

    % Compute ALD for each lag
    for lag = 1:max_lag
        differences = abs(signal(1:end-lag) - signal(1+lag:end));
        differences(differences == 0) = eps;  % Avoid log(0)
        log_divergence = log(differences);
        ald_values(lag) = mean(log_divergence);
    end

    % Plot the ALD values
    figure;
    plot(1:max_lag, ald_values, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    xlabel('Lag');
    ylabel('Average Logarithmic Divergence (ALD)');
    title('Average Logarithmic Divergence vs Lag');
    grid on;

    % Display the final ALD value for the max_lag
    fprintf('Average Logarithmic Divergence (ALD) for lag %d: %f\n', max_lag, ald_values(end));
end
