
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import re 
from sklearn.feature_selection import SelectKBest,SelectPercentile,chi2
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, under_sampling #after split because it will can be enter data leakage
# SMOTE only works with numerical data, so we will use it after transforming the text data into numerical features.
from imblearn.over_sampling import SMOTENC # for categorical data but must have numerical data
from imblearn.over_sampling import SMOTEN # for numerical and categorical data (k_neighbors >= K_samples)
df = pd.read_excel('final_project.ods',engine='odf',dtype=str)
df = df.dropna(axis = 0)
#--------------------------------------
def select_location(location):
    result = re.findall("\,\s[A-Z]{2}$", location)
    if len(result) > 0:
        return result[0][2:]
    else:
        return location
df['location'] = df['location'].apply(select_location)
#--------------------------------------
target = 'career_level'
df = df.dropna(axis=0)
y = df[target]
x = df.drop(columns=target,axis=1)
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
#-------title --------------------------
# tfidf = TfidfVectorizer()
# tfidf.fit(X_train['title'])
# X_train = tfidf.transform(X_train['title']) 
# X_test = tfidf.transform(X_test['title'])
#==================================== imbalanced data ==========================
# cách 1 tăng samples /giảm samples
over_sampler = SMOTEN(sampling_strategy='auto', random_state=42, k_neighbors=2,
                      sampling_strategy_dict=[
                          'director_business_unit_leader' : 50,
                          'managing_director_small_medium_company' : 50,
                          ' specialist' : 20,
                      ])
# có thể chỉ định số lượng sample tăng lên còn lại thì sellect ngẫu nhiên

#cách 2 giảm loss function
# cách 3 dùng thuật toán khác như XGBoost, CatBoost, LightGBM
# cách 4 gộp các features thiểu số thành một class khác 
#!!!!    nếu gộp thì phải gộp theo phân cấp nếu có thể tránh tình trạng phân cách sai lệch 
#!!!!   cách features gộp thì phải có ý nghĩa rõ ràng





#==================================== imbalanced data ==========================


transformer = ColumnTransformer([
    ('title',TfidfVectorizer(stop_words='english'),'title'),  #TF = (số lần từ xuất hiện trong tài liệu) / (tổng số từ trong tài liệu)
                                                                 # IDF = log(Số lượng tài liệu / Số tài liệu chứa từ đó)
    ('location',OneHotEncoder(handle_unknown='ignore'),['location']),
    ('description',TfidfVectorizer(stop_words='english',ngram_range=(1,2),min_df = 0.01,max_df = 0.95),'description'),
    ('function',OneHotEncoder(),['function']),
    ('industry',TfidfVectorizer(stop_words='english'),'industry')])
model = Pipeline([
    ('preprocessing',transformer),
    # ('feature_selection',SelectKBest(chi2,k=1000)),
   ('feature_selection',SelectPercentile(chi2,percentile=50)),
    ('classifier',RandomForestClassifier(n_estimators=100,random_state=42))])
model.fit(X_train,y_train)
pre = model.predict(X_test)
print(classification_report(y_test, pre))


# tang Learning rate rồi mới tối ưu mô hình 
# nếu model nhanh thì có thể có nhiều thời gian tối ưu model 
# giảm đầu vào / giới hạn đầu vào / loại bỏ ngoại lai features ko liên quan



# khi muốn tăng performance !!!! 
# khi tăng / hoặc giảm features / samples tối thiểu không làm cho mô hình giảm performance 
# khi muốn giảm thời gian train thì giảm features / samples - > giảm tối thiểu performance