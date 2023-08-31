# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries which are used for the program.
   
2.Load the dataset.

3.Check for null data values and duplicate data values in the dataframe.

4.Apply logistic regression and predict the y output.

5.Calculate the confusion,accuracy and classification of the dataset. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: A.Anbuselvam
RegisterNumber:  22009081

import pandas as pd
df=pd.read_csv("Placement_Data(1).csv")
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
![Screenshot 2023-08-31 093622](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/f2419726-3070-4b3a-b301-eb25c3e45865)

![Screenshot 2023-08-31 093631](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/05041c9b-729c-40f3-80f5-005644d31b72)

![Screenshot 2023-08-31 093640](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/ad932671-4226-4378-aced-b2edd9a982ff)

![Screenshot 2023-08-31 093647](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/74e5ddba-4b81-4aa4-be00-f0ffd3487bfb)

![Screenshot 2023-08-31 093703](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/04d8ab37-549b-4ed2-9495-94e257869f95)

![Screenshot 2023-08-31 093717](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/ca274c29-2ba5-4ba6-bc65-ce343654bd51)

![Screenshot 2023-08-31 093730](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/90f64480-be2b-4f33-af75-946f8743dafb)

![Screenshot 2023-08-31 093744](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/e67f531d-309f-4dc2-96e6-f57383652e09)

![Screenshot 2023-08-31 093759](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/a69d8079-3973-48fb-b4ff-20d2e1025646)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
