function ald_value = average_log_divergence(signal, lag)
    % Ensure the signal is a column vector
    signal = signal(:);
    
    % Check if lag is within the valid range
    if lag <= 0 || lag >= length(signal)
        error('Lag must be between 1 and length of the signal - 1');
    end
    
    % Compute the differences between the lagged values
    differences = abs(signal(1:end-lag) - signal(1+lag:end));
    
    % Remove zeros to avoid issues with logarithms
    differences(differences == 0) = eps;
    
    % Calculate logarithmic divergence
    log_divergence = log(differences);
    
    % Compute the average of the logarithmic divergence
    ald_value = mean(log_divergence);

    % Display the result
    fprintf('Average Logarithmic Divergence (ALD) with lag %d: %f\n', lag, ald_value);
end