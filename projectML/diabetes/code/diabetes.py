import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
import tensorflow as tf  
from keras.models import load_model
url = r'C:\Users\97ngu\OneDrive\Desktop\course\ML\projectML\diabetes\csv\diabetes.csv'
df = pd.read_csv(url)
# print(df.head())
# print(df.info()) 
# print(df.describe())
# print(df.columns)#[8 rows x 9 columns]
# print(df.shape) #(768, 9)

#===missing data ====#
# print(df.isnull().sum().sum())
# print(df.isna().sum())
#=== 
X = df.drop(columns='Outcome',axis=0)
y = df['Outcome']
#==========================
# for i in range(len(df.columns[:-1])):
#     label = df.columns[i]
#     plt.hist(df[df['Outcome']==1][label], color='blue', label="Diabetes", alpha=0.7, density=True, bins=15)
#     plt.hist(df[df['Outcome']==0][label], color='red', label="No diabetes", alpha=0.7, density=True, bins=15)
#     plt.title(label)
#     plt.ylabel("Probability")
#     plt.xlabel(label)
#     plt.legend()
#     plt.show()
# print(df[df['Outcome'] ==1].shape) #268
# print(df[df['Outcome'] ==0].shape)#500

#=================================split======
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=42)
# print(X_train.shape,X_test.shape) #537,116,115
# print(y_test.shape,y_test.shape)
#=============STD Scaler =========#
from sklearn.preprocessing import StandardScaler 
scaler_sdt = StandardScaler()
X_train= scaler_sdt.fit_transform(X_train)
X_test = scaler_sdt.transform(X_test)
#=========imbalance
from imblearn.over_sampling import RandomOverSampler
over = RandomOverSampler()
X_train, y_train = over.fit_resample(X_train, y_train)
# print(y[y==1].shape)
# print(y[y==0].shape)
#==================Kford============ because X_train too small ==> Xtrain + Xvail to train model

#==================Kford============ because X_train too small ==> Xtrain + Xvail to train model
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score

#===model 1 
'''
params = {
    'n_estimators' : [50,100,150,200],
    'criterion' : ['gini', 'entropy', 'log_loss']
}
model = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),  #model RDF
    param_grid= params,                            #RDF of params
    scoring='recall',                         #metrics
    cv=5,                    # K_ford validation split
    verbose=1  ,
    n_jobs=5
)

#               precision    recall  f1-score   support
#               precision    recall  f1-score   support


#            0       0.82      0.71      0.76        76
#            1       0.56      0.70      0.62        40

#     accuracy                           0.71       116
#    macro avg       0.69      0.71      0.69       116
# weighted avg       0.73      0.71      0.71       116

model.fit(X_train,y_train)
print(model.score)
y_pred = model.predict(X_test)
# print("Grid Search Scores:")
# for mean_score, params in zip(model.cv_results_['mean_test_score'], model.cv_results_['params']):
#     print(f"Mean Accuracy: {mean_score:.4f} | Parameters: {params}")
print("Best Parameters:", model.best_params_)#Best Parameters: {'criterion': 'gini', 'n_estimators': 200}
print("Best Score:", model.best_score_) #Best Score: 0.9341176470588236
print(classification_report(y_test,y_pred))
# model = RandomForestClassifier(random_state=42,criterion='gini',n_estimators=200)
# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)
# print(classification_report(y_test,y_pred))
'''
'''
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation='relu',input_dim =8  ),
          tf.keras.layers.Dense(16,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
loss=tf.keras.losses.BinaryCrossentropy(),
              metrics = ['accuracy'])
# print(model.evaluate(X_train,y_train))
# print('-'*10)
history = model.fit(X_train,y_train,epochs=20,batch_size = 16,validation_split =0.15)
print(model.evaluate(X_test,y_test))
#[0.5051190853118896, 0.7413793206214905]

model.save('modeldiabetes.h5')
'''


#======== lazy predict ========# 
'''
from lazypredict.Supervised import LazyClassifier
#https://lazypredict.readthedocs.io/en/latest/usage.html#classification
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
'''
# AdaBoostClassifier                 0.79               0.81     0.81      0.80        0.05
# QuadraticDiscriminantAnalysis      0.78               0.78     0.78      0.79        0.01
# GaussianNB                         0.72               0.72     0.72      0.72        0.01
'''
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
grid_search = GridSearchCV(estimator=gnb, param_grid=param_grid, cv=5, scoring='accuracy',verbose=1)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate on test data
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)
# Best Parameters: {'var_smoothing': 1e-09}
# Best Cross-Validation Accuracy: 0.715802297250261
# Test Accuracy: 0.7155172413793
'''

from sklearn.ensemble import AdaBoostClassifier
# Initialize AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=50, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test,y_pred))


#               precision    recall  f1-score   support

#            0       0.89      0.74      0.81        76
#            1       0.62      0.82      0.71        40

#     accuracy                           0.77       116
#    macro avg       0.76      0.78      0.76       116
# weighted avg       0.80      0.77      0.77       116

#tips 
'''
tìm những models phù hợp nhất trong lazypredict 
===? tất cả tham số đều là default 
>>> chú trọng vào time taken, tối ưu tốc độ cùng chất lượng
sao đó đi sâu vào từng models bằng Grid Search CV 
==? tìm ra tham số phù hợp nhấ

sử dụng tham số đó để đưa ra model phù hợp nhất 

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation='relu',input_dim =8  ),
          tf.keras.layers.Dense(16,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
loss=tf.keras.losses.BinaryCrossentropy(),
              metrics = ['accuracy'])
# print(model.evaluate(X_train,y_train))
# print('-'*10)
history = model.fit(X_train,y_train,epochs=20,batch_size = 16,validation_split =0.15)
# print(model.evaluate(X_test,y_test))
model.save('modeldiabetes.h5')
'''