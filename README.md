# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required packages.

2.Import the dataset to operate on.

3.Split the dataset.

4.Predict the required output.

5.End the program.
## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: BALA MURUGAN
RegisterNumber: 212222230017

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


```

## Output:

![243150574-6f7e92e1-fa6d-49db-97d0-d49137816e7a](https://github.com/Bala1511/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118680410/5fbbb386-59e1-4cd9-8a89-a8cc08187bcd)


![243150589-e9e3bea5-7dbd-4180-9958-f70be63dd6b6](https://github.com/Bala1511/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118680410/716ea346-7ebc-4a4c-acfe-b8c14d98824a)


![243150599-c0a3825c-5a8a-4135-a3b6-d8ad263e7faa](https://github.com/Bala1511/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118680410/00887939-41a9-4f9b-8bbf-6a8fdf9cd941)

![243151284-2855afe7-bdb1-4e12-a2d0-e0a957e3eb3a](https://github.com/Bala1511/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118680410/754471a8-01db-4a3c-9b57-4b82e30d520a)

![243151294-5695d1a4-84a5-4f4a-94fd-c20dc3d80805](https://github.com/Bala1511/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118680410/6ab1601f-30bd-45de-a797-c5706463db68)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
