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
Placement data:
![image](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/bf54d092-cf50-400a-961a-650be2151d60)


Salary data:
![Screenshot 2023-10-06 080629](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/ca636049-6b2b-4020-9b5d-3d29bb850151)

null function:
![Screenshot 2023-10-06 080635](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/c9857b05-cb90-4b34-a884-b9350f15c2e9)

duplicate:
![Screenshot 2023-10-06 080642](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/6aaf62f3-84af-4182-8acb-051365b94283)

label encoding:
![Screenshot 2023-10-06 080653](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/99d97405-654e-49db-8648-a234a2f282ee)

X status:
![Screenshot 2023-10-06 080659](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/5a5e32b9-b2c6-4572-8718-076962094364)

Y status:
![Screenshot 2023-10-06 080705](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/cfd69d11-65ba-4a18-8869-26357b436204)

y_prediction array:
![image](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/d68faacd-4d88-4747-949e-bc04ed1556e7)

Accuracy value:
![Screenshot 2023-10-06 080750](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/0fe846e9-1142-40b8-8841-cf5160dcbbaa)

Confusion array:
![Screenshot 2023-10-06 080755](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/476622f9-2471-4b53-8e18-15d123a2a06b)

classification report:
![Screenshot 2023-10-06 080813](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/cf74d9e1-1ccc-4229-8297-acd508cf8c94)

Prediction of LR:
![Screenshot 2023-10-06 080855](https://github.com/anbuselvamA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119559871/f4e16084-ffb6-4ba8-8b2b-4331f5c2e9e8)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
