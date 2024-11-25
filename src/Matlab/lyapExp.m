clear; clc
fs = 30;
dim = 3;

% Folder path where the files are located
folderPath = 'E:\PhD Work (Local)\Sit to Stand Fall Risk\np\';

% Get list of all files in the folder
files = dir(fullfile(folderPath, '*.npy'));

% Preallocate a structure to store the results
numFiles = length(files);
results(numFiles).fileName = '';  % Preallocate the structure array with an empty string for fileName
results(numFiles).lyapunovExponent = [];  % Preallocate with an empty array for lyapunovExponent

for i = 1:length(files)
    % Get the full file name (including path)
    fileName = fullfile(folderPath, files(i).name);

    % Extract the file name without the extension
    [~, nameWithoutExt, ~] = fileparts(files(i).name);
    
    % Display the file being processed
    fprintf('Loading file: %s\n', files(i).name);
    
    % Load the .npy file using Python integration
    try
        numpy = py.importlib.import_module('numpy');  % Import numpy module in MATLAB
        data = numpy.load(fileName);                  % Load the .npy file
        matlabData = double(data);                    % Convert to MATLAB array (assuming it's numeric)
        
        % Process the data
        [~,lag] = phaseSpaceReconstruction(matlabData(:,1),[],dim);
        lyap_Exp = lyapunovExponent(matlabData(:,1),fs,lag,dim,'ExpansionRange',[8, 25]);
        app_Entropy = approximateEntropy(matlabData(:,1), lag, dim);

        % Save the file name (without extension) and Lyapunov exponent in the results structure
        results(i).fileName = nameWithoutExt;  % Store the file name without extension
        results(i).lyapunovExponent = lyap_Exp;  % Store the calculated Lyapunov exponent
        results(i).approxEntropy = app_Entropy;
        
    catch ME
        % If there is an error, display it
        fprintf('Error loading file: %s\n', files(i).name);
        disp(ME.message);
    end
end

% Check for non-scalar or empty Lyapunov exponent values and handle them
for i = 1:numFiles
    if isempty(results(i).lyapunovExponent) || ~isscalar(results(i).lyapunovExponent)
        results(i).lyapunovExponent = NaN;  % Assign NaN to invalid entries
    end
end

% Extract Lyapunov exponent values from the structure
lyapunovValues = [results.lyapunovExponent];  % Extract all Lyapunov exponent values

% Sort the Lyapunov exponents in descending order, ignoring NaN values
[~, sortedIdx] = sort(lyapunovValues, 'descend', 'MissingPlacement', 'last');  % NaN values go to the end

% Rearrange the results structure based on the sorted indices
sortedResults = results(sortedIdx);  % Reorder the structure




