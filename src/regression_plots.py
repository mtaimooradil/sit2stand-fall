import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression

# Load the data
# data = pd.read_csv(r"E:\PhD Work (Local)\Sit to Stand Fall Risk\sit2stand-fall\stats\dataClean.csv")
data = pd.read_csv(r"E:\PhD Work (Local)\Sit to Stand Fall Risk\sit2stand-fall\stats\merged_Data_Stability_Entropy_AutoCorr.csv")

# Split data into two groups based on 'ageGroup'
group_50_80 = data[data['ageGroup'] == '[50-80+']
group_rest = data[data['ageGroup'] != '[50-80+']

selected_params = ['left_knee_max_mean_stand2sit', 'neck_min_y_acc', 'pelvic_avg_speed',
                     'trunk_lean_min_ang_acc', 'right_ankle_max_sd_stand2sit',
                     'left_knee_min_ang_vel_stand2sit', 'time_stand2sit', 'left_hip_min_ang_vel_sit2stand',
                     'right_ankle_max_mean_stand2sit', 'pelvic_avg_y_speed',
                     'left_shank_angle_min_sd_stand2sit', 'right_ankle_ang_acc',
                     'pelvic_avg_y_acc', 'right_hip_max', 'right_hip_min_ang_acc', 'left_shank_angle_ang_acc', 'neck_avg_y_acc',
                     'speed', 'pelvic_min_speed_sit2stand']

stability_params = ['lyapunovExponent', 'approxEntropy', 'sampleEntropy', 'MSE1', 'MSE2', 'MSE3', 'MSE4', 'MSE5', 'Ci', 'half_cycle_regularity', 'full_cycle_regularity', 'symmetry_1', 'symmetry_2']
names = ['Lyapunov Exponent', 'Approximate Entropy', 'Sample Entropy', 'Multiscale Entropy ', 'Multiscale Entropy t=2', 'Multiscale Entropy t=3', 'Multiscale Entropy t=4', 'Multiscale Entropy t=5', 'Complexity Index', 'Half Cycle Regularity', 'Full Cycle Regularity', 'Symmetry 1', 'Symmetry 2']

# Extract Age and input parameters
age = data['Age']
input_params = data[stability_params]#[['alignment_max_ang_acc_sit2stand','alignment_max_ang_acc_stand2sit']]#

age_50_80 = group_50_80['Age']
param_50_80 = group_50_80[stability_params]#[['alignment_max_ang_acc_sit2stand','alignment_max_ang_acc_stand2sit']]#[selected_params]

age_rest = group_rest['Age']
param_rest = group_rest[stability_params]#[['alignment_max_ang_acc_sit2stand','alignment_max_ang_acc_stand2sit']]#[selected_params]

# Function to identify and remove outliers based on Cook's Distance
# def remove_outliers(age, param):
#     X = sm.add_constant(age)
#     model = sm.OLS(param, X).fit()
#     influence = model.get_influence()
#     cooks_d = influence.cooks_distance[0]
#     threshold = 0.01 / len(age)
#     non_outliers = cooks_d < threshold

#     # Reset index for the filtered age and param values
#     age_clean = age[non_outliers].reset_index(drop=True)
#     param_clean = param[non_outliers].reset_index(drop=True)

#     return age_clean, param_clean

# Function to identify and remove outliers based on IQR
def remove_outliers(age, param):
    # Calculate the first (Q1) and third (Q3) quartiles
    Q1 = param.quantile(0.25)
    Q3 = param.quantile(0.75)
    IQR = Q3 - Q1  # Interquartile Range
    
    # Define the outlier cutoff thresholds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out the outliers
    non_outliers = (param >= lower_bound) & (param <= upper_bound)

    # Reset index for the filtered age and param values
    age_clean = age[non_outliers].reset_index(drop=True)
    param_clean = param[non_outliers].reset_index(drop=True)

    return age_clean, param_clean

# Function to compute confidence intervals
def confidence_intervals(age_clean, param_clean):
    X_clean = sm.add_constant(age_clean)
    model = sm.OLS(param_clean, X_clean).fit()
    predictions = model.get_prediction(X_clean)
    summary_frame = predictions.summary_frame(alpha=0.05)  # 95% confidence interval
    ci_lower = summary_frame['mean_ci_lower']
    ci_upper = summary_frame['mean_ci_upper']
    return ci_lower, ci_upper

def regression(age, param):
    age_clean, param_clean = remove_outliers(age, param)
    
    X = np.array(age_clean).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, param_clean)
    param_pred = model.predict(X)
    
    R, p_value = stats.pearsonr(age_clean, param_clean)

    ci_lower, ci_upper = confidence_intervals(age_clean, param_clean)
    
    sorted_indices = np.argsort(age_clean)
    age_sorted = age_clean[sorted_indices]
    param_pred_sorted = param_pred[sorted_indices]
    ci_lower_sorted = ci_lower[sorted_indices]
    ci_upper_sorted = ci_upper[sorted_indices]

    return {
        'age_clean': age_clean,
        'param_clean': param_clean,
        'p_value': p_value,
        'R': R,
        'age_sorted': age_sorted,
        'param_pred_sorted': param_pred_sorted,
        'ci_lower_sorted': ci_lower_sorted,
        'ci_upper_sorted': ci_upper_sorted,
    }

def plot_regression(age, param, folder):
    
    # Perform linear regression
    regression_results = regression(age, param)
    age_clean = regression_results['age_clean']
    param_clean = regression_results['param_clean']
    p_value = regression_results['p_value']
    age_sorted = regression_results['age_sorted']
    param_pred_sorted = regression_results['param_pred_sorted']
    ci_lower_sorted = regression_results['ci_lower_sorted']
    ci_upper_sorted = regression_results['ci_upper_sorted']

    print(p_value)


    if p_value < 0.05:

        # Plot the results
        plt.figure()
        plt.scatter(age_clean, param_clean, color='black', label='Data', s = 10, facecolors='none')

        # Plot the regression line
        plt.plot(age_sorted, param_pred_sorted, color='blue', label='Fitted line')

        # Plot confidence interval lines
        plt.plot(age_sorted, ci_lower_sorted, color='red', linestyle='--', label='95% CI')
        plt.plot(age_sorted, ci_upper_sorted, color='red', linestyle='--')

        # # Display R and p-value on the plot
        # plt.text(0.1 * max(age_clean), 0.9 * max(param_clean),
        #         f'R = {R:.2f}, p = {p_value:.3f}', fontsize=12)
        
        # Set title and labels
        plt.xlabel('Age')
        plt.ylabel(column)

        plt.legend()
        plt.savefig(f"E:/PhD Work (Local)/Sit to Stand Fall Risk/data/regression_plots/{folder}/{column}.png")
        plt.close()


# Loop through each parameter and perform linear regression
# for column in input_params.columns:  
#     param = input_params[column]
#     plot_regression(age, param, 'all')

# for column in param_50_80.columns:  
#     param = param_50_80[column]
#     plot_regression(age_50_80, param, '50_80')

# for column in param_rest.columns:  
#     param = param_rest[column]
#     plot_regression(age_rest, param, 'rest')


def plot_regression_multiple(age_all, param_all, age_50_80, param_50_80, folder, column):

    # Perform linear regression for all data
    regression_all = regression(age_all, param_all)
    regression_50_80 = regression(age_50_80, param_50_80)

    if True: #regression_50_80["p_value"] < 0.05:

        # Scatter plot for both groups
        plt.figure(figsize=(10 / 2.54, 8 / 2.54))
        
        # Scatter plot for all data
        plt.scatter(regression_all['age_clean'], regression_all['param_clean'], color='black', label=f'Age < 50 R={regression_all["R"]:.3f}', s=7, facecolors='none')

        # Plot the regression line for all data
        plt.plot(regression_all['age_sorted'], regression_all['param_pred_sorted'], color='blue')

        # Plot confidence interval lines for all data
        plt.plot(regression_all['age_sorted'], regression_all['ci_lower_sorted'], color='blue', linestyle='--')
        plt.plot(regression_all['age_sorted'], regression_all['ci_upper_sorted'], color='blue', linestyle='--')

        # Scatter plot for 50-80+ data
        plt.scatter(regression_50_80['age_clean'], regression_50_80['param_clean'], color='green', label=f'Age > 50 R={regression_50_80["R"]:.3f}', s=7, facecolors='none')

        # Plot the regression line for 50-80+ data
        plt.plot(regression_50_80['age_sorted'], regression_50_80['param_pred_sorted'], color='red')

        # Plot confidence interval lines for 50-80+ data
        plt.plot(regression_50_80['age_sorted'], regression_50_80['ci_lower_sorted'], color='red', linestyle='--')
        plt.plot(regression_50_80['age_sorted'], regression_50_80['ci_upper_sorted'], color='red', linestyle='--')

        # Set title and labels
        #y_label = column.replace('_', ' ').title()
        y_label = names[stability_params.index(column)]
        plt.xticks(np.arange(15, 100, 10), fontsize=9)
        plt.yticks(fontsize=9)
        plt.xlabel('Age', fontsize=10, fontweight='bold', fontname='Arial')
        plt.ylabel(y_label, fontsize=10, fontweight='bold', fontname='Arial')
        plt.legend(fontsize=9, frameon=True)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        # plt.title(f'(R = {regression_all["R"]:.3f}, p = {regression_all["p_value"]:.3f}) vs (R = {regression_50_80["R"]:.3f}, p = {regression_50_80["p_value"]:.3f})')
        # plt.title(f'(R = {regression_all["R"]:.3f}) vs (R = {regression_50_80["R"]:.3f})')
        plt.tight_layout()
        plt.savefig(f"E:/PhD Work (Local)/Sit to Stand Fall Risk/data/regression_plots/{folder}/{column}.png")
        plt.close()



# Loop through each parameter and perform regression comparison
# for column in input_params.columns:
#     # Ensure the column exists in param_50_80 and param_rest
#     if column in param_50_80.columns:
#         param_all = input_params[column]
#         param_50_80_col = param_50_80[column]
        
#         # Plot the regression for the full dataset and 50-80+ age group
#         plot_regression_multiple(age, param_all, age_50_80, param_50_80_col, 'comparison', column)
#     else:
#         print(f"Skipping column '{column}' as it is not present in the 50-80+ group.")

# Loop through each parameter and perform regression comparison
for column in input_params.columns:

    try:
        # Ensure the column exists in param_50_80 and param_rest
        if column in param_rest.columns:
            param_all = param_rest[column]
            param_50_80_col = param_50_80[column]
            
            # Plot the regression for the full dataset and 50-80+ age group
            plot_regression_multiple(age_rest, param_all, age_50_80, param_50_80_col, 'stability', column)
        else:
            print(f"Skipping column '{column}' as it is not present in the 50-80+ group.")
    except:
        print(f"Error in column '{column}'")