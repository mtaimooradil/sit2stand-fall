import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname('src'), '..'))
from utils import *
import scipy.io

# Load the data
data = np.load('time_series_all.npy', allow_pickle=True).item()

# Loop through all subjects
for sub in data.keys():
    
    if sub in problematic_subjects:
        # For problematic subjects, process NaNs and trim the data
        indices = np.int_(np.loadtxt(f'E:\PhD Work (Local)\Sit to Stand Fall Risk\sit2stand-fall\{sub}.txt')[:,0])
        data[sub]['data'][indices,:,1] = np.nan
        
        # Process each joint's time series to handle NaNs at the start or end
        for i in range(data[sub]['data'].shape[1]):  # assuming second axis is for the joints/positions
            series_with_nans = pd.Series(data[sub]['data'][:, i, 1])

            # Find the index of the first and last non-NaN values
            first_valid_index = series_with_nans.first_valid_index()
            last_valid_index = series_with_nans.last_valid_index()

            # Slice the series to remove the NaNs at the start and end
            trimmed_series = series_with_nans[first_valid_index:last_valid_index + 1]

            # Interpolate missing NaNs inside the valid range
            interpolated_series = trimmed_series.interpolate(method='linear', limit_area='inside')

            # Replace the original data with the trimmed and interpolated series
            data[sub]['data'][:, i, 1] = np.nan  # Set all to NaN initially
            data[sub]['data'][first_valid_index:last_valid_index + 1, i, 1] = interpolated_series.values

        # Store the trimmed data for problematic subjects
        data[sub]['trimmed_data'] = data[sub]['data'][first_valid_index:last_valid_index + 1]

    else:
        # For non-problematic subjects, create trimmed_data with the full original data
        data[sub]['trimmed_data'] = data[sub]['data']  # Copy the original unmodified data