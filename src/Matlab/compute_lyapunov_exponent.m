function lyapunov_exp = compute_lyapunov_exponent(signal, embedding_dim, time_delay, num_neighbors)
    % Compute the largest Lyapunov exponent of a time series
    %
    % signal         - Input time series (vector)
    % embedding_dim  - Embedding dimension for phase space reconstruction
    % time_delay     - Time delay for phase space reconstruction
    % num_neighbors  - Number of nearest neighbors to track divergence

    % Phase space reconstruction using Takens' embedding theorem
    N = length(signal) - (embedding_dim - 1) * time_delay;
    
    if N <= 0
        error('The combination of embedding dimension and time delay is too large for the given signal length.');
    end

    reconstructed = zeros(N, embedding_dim);
    for i = 1:embedding_dim
        reconstructed(:, i) = signal((1:N) + (i - 1) * time_delay);
    end

    % Compute nearest neighbors using Euclidean distance
    distances = pdist2(reconstructed, reconstructed);
    distances(distances == 0) = Inf;  % Avoid self-matching
    [sorted_distances, indices] = sort(distances, 2);

    % Track divergence over time
    valid_indices = 1:(N - time_delay);  % Ensure no index exceeds the bounds
    divergence = zeros(length(valid_indices), num_neighbors);

    for j = 1:num_neighbors
        neighbor_idx = indices(:, j);
        for i = valid_indices
            if neighbor_idx(i) + time_delay <= N
                divergence(i, j) = norm(reconstructed(i + time_delay, :) - reconstructed(neighbor_idx(i) + time_delay, :));
            else
                divergence(i, j) = NaN;  % Handle out-of-bound indices safely
            end
        end
    end

    % Remove NaNs and compute average divergence
    valid_divergence = divergence(~isnan(divergence));
    avg_divergence = mean(valid_divergence)

    % Compute the Lyapunov exponent as the slope of log divergence
    lyapunov_exp = mean(log(avg_divergence));

    % Plot the divergence curve
    figure;
    plot(log(valid_divergence), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    xlabel('Time Index');
    ylabel('Logarithm of Divergence');
    title('Divergence Curve for Lyapunov Exponent Estimation');
    grid on;

    % Display the estimated Lyapunov exponent
    fprintf('Estimated Largest Lyapunov Exponent: %f\n', lyapunov_exp);
end
