import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd 
import plotly.express as px 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df=pd.read_csv("labels.csv")
y=df["labels"]
print(len(y))
X=np.load("image.npz")["arr_0"]
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=8000,test_size=2000)
X_train=X_train/255.0
X_test=X_test/255.0
clf=LogisticRegression()
clf.fit(X_train,y_train)
y_predict=clf.predict(X_test)
accuracy=accuracy_score(y_predict,y_test)
print(accuracy)
cm=confusion_matrix(y_predict,y_test)
fig=px.scatter(x=[1,2,3,4,5],y=[1,2,3,4,5])
fig.show()
# print(cm)
sns.heatmap(cm)
plot=plt.figure(figsize=(12,12))
plot=sns.heatmap(cm,annot=True)
plt.show()