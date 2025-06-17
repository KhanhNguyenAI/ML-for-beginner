import pandas as pd 
import tensorflow as tf  
from keras.models import load_model


model = load_model(r'C:\Users\97ngu\OneDrive\Desktop\course\ML\projectML\diabetes\model\modeldiabetes.h5')
loss,acc=model.evaluate(X_test,y_test)
X_new = X_test[10]
y_new = y_test[10]
# print(X_new)                #1D
X_new = np.expand_dims(X_new,axis=0)           #2D
# print(X_new)

y_pre = model.predict(X_new)
# print(y_pre)
# print(y_new)
if y_pre <= 0.5 : 
    print('no')
else : 
    print('Yes')
