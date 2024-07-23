# Importing the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

TARGET = 'fallsBin'
TEST_RATIO = 0.2
RANDOM_STATE = 42

# Data Loading
data = pd.read_csv("./stats/dataClean v2.csv")

# Converting Martial Status to Single and Not Single (due to the small number of samples in the other categories)
# Single -> 0, Not Single -> 1
data['Marital'] = data['Marital'].apply(lambda x: 1 if x in [1, 6] else 0)

# For Education, a value of 5 mean prefer not to answer so change these to 0
# Change 2 to 1 (High School), 3 to 2 (College) and 4 to 3 (Higher)
data['Education'] = data['Education'].apply(lambda x: 0 if x == 5 else x-1 if x in [2, 3, 4] else x)
# Converting Education to One Hot Encoding
encoder = OneHotEncoder()
encoded_education = encoder.fit_transform(data[['Education']])
data_encoded = pd.concat([data.drop(['Education'], axis=1), pd.DataFrame(encoded_education.toarray(), columns=encoder.get_feature_names_out())], axis=1)
data = data_encoded

# Converting Ethinicity to White, Asians and Others
# White -> 1, Asian -> 2, Others -> 3
data['EthCat'] = data['EthCat'].apply(lambda x: 1 if x == 1 else 2 if x == 2 else 3)
# Converting Ethnicity to One Hot Encoding
encoded_ethnicity = encoder.fit_transform(data[['EthCat']])
data_encoded = pd.concat([data.drop(['EthCat'], axis=1), pd.DataFrame(encoded_ethnicity.toarray(), columns=encoder.get_feature_names_out())], axis=1)
data = data_encoded

# For Employment, a value of 10 mean prefer not to answer so change these to 0
data['Employment'] = data['Employment'].apply(lambda x: 0 if x == 10 else x)
# Converting Employment to One Hot Encoding
encoded_employment = encoder.fit_transform(data[['Employment']])
data_encoded = pd.concat([data.drop(['Employment'], axis=1), pd.DataFrame(encoded_employment.toarray(), columns=encoder.get_feature_names_out())], axis=1)
data = data_encoded

# Splitting the data
predictors = data.drop(TARGET, axis=1)
response = data[TARGET]
predictors_train, predictors_test, response_train, response_test = train_test_split(predictors, 
                                                                                    response, 
                                                                                    test_size=TEST_RATIO, 
                                                                                    stratify=response, 
                                                                                    random_state=RANDOM_STATE)

# Standardizing/normalizing the data
std_scalar = StandardScaler()
minmax_scalar = MinMaxScaler()

# Exclude all categorical columns from Standardizing/normalizing
categorical_columns = [
'Sex',
'Income',
'Marital',
'OA_check',
'stress',
'PA_cat',
'numMedCond',
'backpain',
'depression',
'highBP',
'falling_1',
'falling_2',
'falling_3',
'Education_0',
'Education_1',
'Education_2',
'Education_3',
'EthCat_1',
'EthCat_2',
'EthCat_3',
'Employment_0',
'Employment_1',
'Employment_2',
'Employment_3',
'Employment_4',
'Employment_5',
'Employment_6',
'Employment_7',
'Employment_8'
]

def scaling_data(data_train, data_test, scaling_function, exclude = []):
    predictors_train_to_scale = data_train.drop(exclude, axis=1)
    predictors_test_to_scale = data_test.drop(exclude, axis=1)

    predictors_train_scaled = scaling_function.fit_transform(predictors_train_to_scale)
    predictors_test_scaled = scaling_function.transform(predictors_test_to_scale)

    predictors_train_scaled_df = pd.DataFrame(predictors_train_scaled, columns=predictors_train_to_scale.columns, index=predictors_train_to_scale.index)
    predictors_test_scaled_df = pd.DataFrame(predictors_test_scaled, columns=predictors_test_to_scale.columns, index=predictors_test_to_scale.index)

    predictors_train_scaled_df = pd.concat([predictors_train_scaled_df, data_train[exclude]], axis=1)
    predictors_test_scaled_df = pd.concat([predictors_test_scaled_df, data_test[exclude]], axis=1)
    
    return predictors_train_scaled_df, predictors_test_scaled_df

predictors_train_standardized_df, predictors_test_standardized_df = scaling_data(predictors_train.copy(), predictors_test.copy(), StandardScaler(), exclude=categorical_columns)
predictors_train_normalized_df, predictors_test_normalized_df = scaling_data(predictors_train.copy(), predictors_test.copy(), MinMaxScaler(), exclude=categorical_columns)


# Models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
# from skorch import NeuralNetClassifier

classifiers = {

    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': XGBClassifier(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Gaussian Process': GaussianProcessClassifier(),
    #'Neural Network': NeuralNetClassifier()
}

# Training the models

classifiers['Logistic Regression'].fit(predictors_train_standardized_df, response_train)
predictions = classifiers['Logistic Regression'].predict(predictors_test_standardized_df)

def training_and_prediction(classifier, predictors_train, response_train, predictors_test):
    classifier.fit(predictors_train, response_train)
    return classifier.predict(predictors_test)

predictions_standardized = {
    name: training_and_prediction(classifier, predictors_train_standardized_df, response_train, predictors_test_standardized_df) 
    for 
    name, classifier 
    in 
    classifiers.items()
    }


predictions_normalized = {
    name: training_and_prediction(classifier, predictors_train_normalized_df, response_train, predictors_test_normalized_df) 
    for 
    name, classifier 
    in 
    classifiers.items()
    }

# Evaluating the models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(predictions, response_test):
    return {
        'accuracy': accuracy_score(response_test, predictions),
        'precision': precision_score(response_test, predictions),
        'recall': recall_score(response_test, predictions),
        'f1': f1_score(response_test, predictions),
        'roc_auc': roc_auc_score(response_test, predictions)
    }

evaluation_standardized = {
    name: evaluate_model(predictions, response_test) 
    for 
    name, predictions 
    in 
    predictions_standardized.items()
}

evaluation_normalized = {
    name: evaluate_model(predictions, response_test) 
    for 
    name, predictions 
    in 
    predictions_normalized.items()
}

# ROC Curve
from sklearn.metrics import roc_curve

def roc_curve_model(classifier, predictors_test, response_test):
    return roc_curve(response_test, classifier.predict_proba(predictors_test)[:,1])

try:
    roc_curve_standardized = {
        name: roc_curve_model(classifier, predictors_test_standardized_df, response_test) 
        for 
        name, classifier 
        in 
        classifiers.items()
    }

    roc_curve_normalized = {
        name: roc_curve_model(classifier, predictors_test_normalized_df, response_test) 
        for 
        name, classifier 
        in 
        classifiers.items()
    }
except:
    print('ROC Curve not available for this model')


# Parameters for GridSearchCV

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


parameters = {
    lr: {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    },
    rf: {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    gb: {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 1]
    },
    svc: {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    },
    dt: {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
}


kf = StratifiedKFold(n_splits=5, shuffle=False)


# GridSearchCV
grid_lr = GridSearchCV(classifiers['Logistic Regression'], param_grid=parameters['lr'], cv=kf, 
                          scoring='recall').fit(X_train, y_train)