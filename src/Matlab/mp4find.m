% Folder paths (modify with your actual folder paths)
folder_npy = 'E:\PhD Work (Local)\Sit to Stand Fall Risk\np\';  % Folder containing .npy files
folder_mp4 = 'E:\PhD Work (Local)\Sit to Stand Fall Risk\Videos\';  % Folder containing .mp4 files

% Get the list of .npy files in the folder
npy_files = dir(fullfile(folder_npy, '*.npy'));
npy_filenames = arrayfun(@(x) erase(x.name, '.npy'), npy_files, 'UniformOutput', false);

% Get the list of .mp4 files in the folder
mp4_files = dir(fullfile(folder_mp4, '*.mp4'));
mp4_filenames = arrayfun(@(x) erase(x.name, '.mp4'), mp4_files, 'UniformOutput', false);

% Find the common filenames between the two folders
common_filenames = intersect(npy_filenames, mp4_filenames);

% Check if the fileName in sortedResults is in the common_filenames list and
% assign 1 if it is found, 0 otherwise
for i = 1:length(sortedResults)
    % Convert fileName to a cell to compare with common_filenames (which is a cell array)
    if ismember(sortedResults(i).fileName, common_filenames)
        sortedResults(i).fileFound = 1;  % File is found (yes)
    else
        sortedResults(i).fileFound = 0;  % File is not found (no)
    end
end

% Display the sortedResults structure with the new 'fileFound' field
for i = 1:length(sortedResults)
    fprintf('File: %s, Lyapunov Exponent: %.4f, File Found: %d\n', ...
        sortedResults(i).fileName, sortedResults(i).lyapunovExponent, sortedResults(i).fileFound);
end

% Add a new field 'fileFound' to sortedResults
% for i = 1:length(sortedResults)
%     % Check if the current file name (from sortedResults) exists in the .mp4 folder
%     if any(strcmp(sortedResults(i).fileName, mp4FileNames))
%         sortedResults(i).fileFound = 1;  % File is found
%     else
%         sortedResults(i).fileFound = 0;  % File is not found
%     end
% end
% 
% % Display the results to verify
% for i = 1:length(sortedResults)
%     fprintf('File: %s, Lyapunov Exponent: %.4f, File Found: %d\n', ...
%         sortedResults(i).fileName, sortedResults(i).lyapunovExponent, sortedResults(i).fileFound);
% end

