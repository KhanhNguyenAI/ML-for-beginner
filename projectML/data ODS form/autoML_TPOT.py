# Cài đặt TPOT nếu chưa có
# pip install tpot
import pandas as pd 
import warnings
import cudf
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Tải dữ liệu mẫu
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, train_size=0.75, random_state=42
)

tpot=TPOTClassifier( generations=3,
   population_size=3,
   config_dict="TPOT cuML",
   scoring='roc_auc',
   max_time_mins=40,
   cv=2,
   verbosity=2)

tpot.fit(x_train.to_numpy(), y_train.to_numpy())
print(tpot.score(x_test.to_numpy(), y_test.to_numpy()))
