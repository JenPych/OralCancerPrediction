import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc
import time
import pickle


# READ CSV FILE
df = pd.read_csv('oral_cancer_prediction_dataset.csv')

# adjust settings for better data display
pd.set_option('display.max_columns', 30)  # display all columns
pd.set_option('display.width', None)  # avoid line space

# CREATE SAVE LOCATION to SAVE GRAPHS
save_dir = ('/Users/jayanshrestha/Downloads/Python_Scripts/MachineLearningAgain/OralCancer/oral_cancer_pred_graphs')

# DATA PROFILING AND ANALYSIS
print("Data information: ")
print(df.info())
print("Data statistics of numerical values:")
print(df.describe())
print("Data frequency of categorical data:")
print(df.describe(include='object'))

# CONFIRMING  NON-EXISTENCE OF NaN VALUES
print("Null Values:")
print(df.isna().sum())

# DROP ID COLUMN AS IT IS NOT NEEDED
df.drop('ID', inplace=True, axis=1)

# EXPLORATORY DATA ANALYSIS USING GRAPHS of all the columns

cat_columns = ['Country', 'Gender', 'Tobacco Use', 'Alcohol Consumption', 'HPV Infection', 'Betel Quid Use',
               'Chronic Sun Exposure', 'Poor Oral Hygiene', 'Diet (Fruits & Vegetables Intake)',
               'Family History of Cancer', 'Compromised Immune System', 'Oral Lesions', 'Unexplained Bleeding',
               'Difficulty Swallowing', 'White or Red Patches in Mouth', 'Treatment Type', 'Early Diagnosis']

num_columns = [col for col in df.columns if col not in cat_columns and col != 'Oral Cancer (Diagnosis)']

for cols in cat_columns:
    plt.figure(figsize=(10, 5))
    counter = sns.countplot(data=df, x=cols, hue='Oral Cancer (Diagnosis)', palette='coolwarm')
    counter.bar_label(counter.containers[0], fontsize=7)
    counter.bar_label(counter.containers[1], fontsize=7)
    plt.xticks(rotation=45)
    plt.title(f'Countplot of {cols}')
    plt.savefig(os.path.join(save_dir, cols))
    # print(plt.show())
    plt.close()

for cols in num_columns:
    plt.figure(figsize=(10, 5))
    hist = sns.histplot(data=df, x=cols, hue='Oral Cancer (Diagnosis)', palette='coolwarm', kde=True)
    plt.title(f'Histogram plot of {cols}')
    plt.savefig(os.path.join(save_dir, cols))
    # print(plt.show())
    plt.close()

sns.heatmap(data=df.corr(numeric_only=True), annot=True, fmt='.2f')
plt.title("Correlation of numerical data")
plt.savefig(os.path.join(save_dir, "Correlation.png"))
# print(plt.show())
plt.close()

# DISTINGUISHING PRE- DIAGNOSIS DATA and POST- DIAGNOSIS DATA
pre_diag = ['Gender', 'Tobacco Use', 'Alcohol Consumption', 'HPV Infection', 'Betel Quid Use', 'Chronic Sun Exposure',
            'Poor Oral Hygiene', 'Family History of Cancer', 'Compromised Immune System', 'Oral Lesions',
            'Unexplained Bleeding', 'Difficulty Swallowing', 'White or Red Patches in Mouth']

post_diag = ['Tumor Size (cm)', 'Cancer Stage', 'Survival Rate (5-Year, %)',
             'Cost of Treatment (USD)','Economic Burden (Lost Workdays per Year)']

# LABEL ENCODING TO CONVERT CATEGORICAL DATA INTO NUMERICAL DATA
le_encoder = LabelEncoder()

# target variables label encoding
df['Early Diagnosis'] = le_encoder.fit_transform(df['Early Diagnosis'])
df['Oral Cancer (Diagnosis)'] = le_encoder.fit_transform(df['Oral Cancer (Diagnosis)'])

# binary features label encoding
for pre_cols in pre_diag:
    df[pre_cols] = le_encoder.fit_transform(df[pre_cols])

# multi features ordinal encoding

print(df['Diet (Fruits & Vegetables Intake)'].unique())
diet_order = ['Low', 'Moderate', 'High']
multi_encoder_diet = OrdinalEncoder(categories=[diet_order])
df['Diet (Fruits & Vegetables Intake)'] = multi_encoder_diet.fit_transform(df[['Diet (Fruits & Vegetables Intake)']])

print(df['Treatment Type'].unique())
treatment_type = ['No Treatment', 'Surgery', 'Radiation', 'Targeted Therapy', 'Chemotherapy']
multi_encoder_treatment = OrdinalEncoder(categories=[treatment_type])
df['Treatment Type'] = multi_encoder_treatment.fit_transform(df[['Treatment Type']])

# FEATURE SELECTION
# this is for pre diagnosis prediction
pre_X = df.loc[:, ['Age', 'Gender', 'Tobacco Use', 'Alcohol Consumption', 'HPV Infection', 'Betel Quid Use',
                   'Chronic Sun Exposure', 'Poor Oral Hygiene', 'Family History of Cancer', 'Compromised Immune System',
                   'Oral Lesions','Unexplained Bleeding', 'Difficulty Swallowing', 'White or Red Patches in Mouth',
                   'Diet (Fruits & Vegetables Intake)']]

pre_y = df['Early Diagnosis']
post_y = df['Oral Cancer (Diagnosis)']

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(pre_X, post_y, test_size=0.2, random_state=42)

# STANDARD SCALING
scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# ALGORITHM SELECTION
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(max_depth=10, n_estimators=50, n_jobs=-1, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

trained_models = {}
result = []


# PIPELINE AND TRAINING
for model_name, model in models.items():
    start_time = time.time()
    if model_name == 'XGBoost':
        param_grid = {
            'eta': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'n_estimators': [50, 100, 200]
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring="f1")
        model_pipeline = Pipeline([('scaler', StandardScaler()), ('model', grid_search)])
        model_pipeline.fit(X_train, y_train)
        best_model = model_pipeline.named_steps['model'].best_estimator_
    else:
        model_pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
        model_pipeline.fit(X_train, y_train)
        best_model = model_pipeline

    trained_models[model_name] = best_model
    prediction_start_time = time.time()
    y_pred = best_model.predict(X_test)
    end_time = time.time()

    fit_time = prediction_start_time - start_time
    prediction_time = end_time - prediction_start_time

    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    precision = precision_score(y_true= y_test, y_pred= y_pred)
    recall = recall_score(y_true=y_test, y_pred= y_pred)
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

# Confusion Matrix Diagram
    plt.figure()
    sns.heatmap(data=cm, annot=True, cmap='coolwarm', fmt='.2f')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix using {model}")
    plt.savefig(os.path.join(save_dir, f"Confusion Matrix using {model.__class__.__name__}"))
    print(plt.show())
    plt.close()

    if model_name == 'XGBoost':
        fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({model_name})')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(save_dir, f"{model_name}_ROC_Curve.png"))
        plt.close()

        feature_importance = best_model.feature_importances_
        feature_names = pre_X.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'{model_name} Feature Importance')
        plt.savefig(os.path.join(save_dir, f"{model_name}_Feature_Importance.png"))
        plt.close()

    result.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'Fit Time (s)': fit_time,
        'Prediction Time (s)': prediction_time
    })

results_df = pd.DataFrame(result)
print(results_df)

# Pickle the trained models
with open('oral_cancer_trained_models.pkl', 'wb') as f:
    pickle.dump(trained_models, f)

