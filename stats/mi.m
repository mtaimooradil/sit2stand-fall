function I = mi(x, y, nbins)
    % MUTUALINFO Compute the mutual information between two variables.
    %   mi = MUTUALINFO(x, y, nbins) computes the mutual information between
    %   the time series x and y using nbins bins for the histogram estimation.
    %
    %   Input:
    %       x     - First time series (vector)
    %       y     - Second time series (vector)
    %       nbins - Number of bins for histogram (scalar)
    %
    %   Output:
    %       mi    - Mutual information (scalar)

    if nargin < 3
        nbins = 64; % Default number of bins
    end
    
    % Remove NaNs if any
    validIdx = ~isnan(x) & ~isnan(y);
    x = x(validIdx);
    y = y(validIdx);

    % Compute the joint histogram
    [pxy, ~, ~] = histcounts2(x, y, nbins, 'Normalization', 'probability');

    % Compute the marginal histograms
    px = sum(pxy, 2); % Sum over columns to get marginal probabilities of x
    py = sum(pxy, 1); % Sum over rows to get marginal probabilities of y

    % Compute the entropies
    Hx = -sum(px .* log(px + eps));
    Hy = -sum(py .* log(py + eps));
    Hxy = -sum(pxy(:) .* log(pxy(:) + eps));

    % Mutual Information
    I = Hx + Hy - Hxy;
end


