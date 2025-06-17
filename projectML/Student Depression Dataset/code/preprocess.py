#from ydata_profiling import ProfileReport
# profile = ProfileReport(df, title = 'student_depression_report')
# profile.to_file("student_depression_report.html")
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import re
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from lazypredict.Supervised import LazyClassifier
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv(r'C:\Users\97ngu\OneDrive\Desktop\course\ML\projectML\Student Depression Dataset\csv\student_depression_dataset.csv')

boolen_f =['Have you ever had suicidal thoughts ?','Family History of Mental Illness'] #1
numeric_features = ['Gender','Profession','Dietary Habits','Degree','Have you ever had suicidal thoughts ?','Family History of Mental Illness','Financial Stress']
normalize_f = ['Sleep Duration'] #2

target = 'Depression'
y = df[target]
x = df.drop(columns=['Depression','id','City'],axis=1)

x = x[x["Sleep Duration"] != "Others"]
x = x[x["Dietary Habits"] != "Others"]
print(x['Dietary Habits'].unique())
## boolen-f 
label_en = LabelEncoder()
for col in numeric_features :
    x[col] = label_en.fit_transform(x[col])
# x['Have you ever had suicidal thoughts ?'] = label_en.fit_transform(x['Have you ever had suicidal thoughts ?'])
# x['Family History of Mental Illness'] = label_en.fit_transform(x['Family History of Mental Illness'])
##nomalize feature
def extract_hours(s):
    # Find a number (including decimals)
    match = re.search(r"(\d+(\.\d+)?)", str(s))
    return float(match.group(1)) if match else np.nan

x["Sleep Duration"] = x["Sleep Duration"].apply(extract_hours)
# Convert Financial Stress to categorical if it represents levels (e.g., Low, Medium, High)



#plt.figure(figsize=(10,8))
# num_cols = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 
#             'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration', 'Work/Study Hours']
# corr_matrix = x[num_cols].corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Heatmap")
# plt.show()
'''
plt.figure(figsize=(8,5))
sns.countplot(x=y, data=x, palette="viridis")
plt.title("Distribution of Depression among Students")
plt.xlabel("Depression (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.legend()
plt.show()
num_features = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 
                'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration', 'Work/Study Hours']
x[num_features].boxplot( figsize=(15,10))
plt.tight_layout()
plt.show()
'''

x = x.dropna()
x = x.drop(columns=['Profession','Degree'])
y = y.loc[x.index]  # Align y with x after dropping NaN values
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state =123)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,test_size=0.5,random_state =123)
'''
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
NearestCentroid                    0.83               0.83     0.83      0.83        0.02
GaussianNB                         0.81               0.82     0.82      0.81        0.02
'''
gnb = GaussianNB()

'''
# Define parameter grid
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=gnb, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate on the test set
test_score = grid_search.score(X_val, y_val)
print("Test Set Accuracy:", test_score)'''
# gnb = GaussianNB(var_smoothing=1e-5)
# gnb.fit(X_train,y_train)
# y_pre = gnb.predict(X_val)
'''
              precision    recall  f1-score   support

           0       0.76      0.87      0.81      1168
           1       0.90      0.80      0.85      1620

    accuracy                           0.83      2788
   macro avg       0.83      0.84      0.83      2788
weighted avg       0.84      0.83      0.83      2788
'''

#===============================================
from xgboost import  XGBClassifier
'''
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.9, 1],
}

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
print('cross validation accuracy')
print(grid_search.best_score_ )
test_score = grid_search.score(X_val,y_val)
print('accuracy')
print(test_score)'''
#{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.7}
xgb = XGBClassifier(learning_rate = 0.1, max_depth= 3, n_estimators= 100, subsample= 0.7)
xgb.fit(X_train,y_train)
y_pre = xgb.predict(X_val)
# print(classification_report(y_val,y_pre))
'''
              precision    recall  f1-score   support

           0       0.82      0.77      0.80      1168
           1       0.84      0.88      0.86      1620

    accuracy                           0.83      2788
   macro avg       0.83      0.83      0.83      2788
weighted avg       0.83      0.83      0.83      2788'''
# import joblib

# Lưu mô hình
# joblib.dump(xgb, 'testAPPmodelStudentDepression.pkl')
print(x['Dietary Habits'].unique())

#### model app basic 
