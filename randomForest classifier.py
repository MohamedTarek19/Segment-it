import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler, PolynomialFeatures
import csv
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier , BaggingClassifier
# from sklearn.svm import SVC

Data = pd.read_csv("train.csv", encoding='unicode_escape')

# print(Data.isnull().sum())
Data = Data.drop(["ID"], axis=1)
data = Data.T

for i in data:
    if data[i].isnull().sum() >= 2:
        data = data.drop([i], axis=1)

data = data.T
data['Profession'] = data['Profession'].fillna(data['Profession'].mode()[0])
data['Ever_Married'] = data['Ever_Married'].fillna(data['Ever_Married'].mode()[0])
data['Work_Experience'] = data['Work_Experience'].fillna(data['Work_Experience'].median())
data["Graduated"] = data["Graduated"].fillna(data["Graduated"].mode()[0])
data['Family_Size'] = data['Family_Size'].fillna(int(data['Family_Size'].mean()))
data["Var_1"] = data["Var_1"].fillna(data["Var_1"].mode()[0])
print(data.isnull().sum())

for j in range(0, len(data["Age"])):
    if data.iat[j, 2] <= 20:
        data.iat[j, 3] = "No"
label = LabelEncoder()
data["Gender"] = label.fit_transform(data["Gender"])
data["Ever_Married"] = label.fit_transform(data["Ever_Married"])
data["Graduated"] = label.fit_transform(data["Graduated"])
data["Profession"] = label.fit_transform(data["Profession"])
data["Spending_Score"] = label.fit_transform(data["Spending_Score"])
data["Var_1"] = label.fit_transform(data["Var_1"])

X = data.drop(["Segmentation"], axis=1).values
Y = data["Segmentation"]

# rfc = RandomForestClassifier()

# forest_params = [{'max_depth': list(range(10, 15)),
#                   'n_estimators': list(range(80, 150))}]

# clf = GridSearchCV(rfc, forest_params, cv = 5, scoring='accuracy')

# clf.fit(X, Y)

# print(clf.best_params_)

# print(clf.best_score_)
#dtree = tree.DecisionTreeClassifier(random_state=40, max_depth=4, criterion="gini", min_samples_leaf=1)
#dtree.fit(X, Y)
# bgclass = BaggingClassifier(n_estimators= 120,max_features= 8,random_state=0)
RForest= RandomForestClassifier(n_estimators=115,criterion='gini', 
                                max_depth= 17,
                                min_samples_split=50, min_samples_leaf=2,max_features= "auto" , random_state=0)
#, min_samples_split=50, min_samples_leaf=5,
RForest.fit(X,Y)



##############################[test data]##################################

test_Data = pd.read_csv("test.csv", encoding='unicode_escape')

print(test_Data.isnull().sum())
test_data = test_Data
test_data['Profession'] = test_data['Profession'].fillna(test_data['Profession'].mode()[0])
test_data['Ever_Married'] = test_data['Ever_Married'].fillna(test_data['Ever_Married'].mode()[0])
test_data['Work_Experience'] = test_data['Work_Experience'].fillna(test_data['Work_Experience'].median())
test_data["Graduated"] = test_data["Graduated"].fillna(test_data["Graduated"].mode()[0])
test_data['Family_Size'] = test_data['Family_Size'].fillna(int(test_data['Family_Size'].mean()))
test_data["Var_1"] = test_data["Var_1"].fillna(test_data["Var_1"].mode()[0])
print(test_data.isnull().sum())

label = LabelEncoder()
test_data["Gender"] = label.fit_transform(test_data["Gender"])
test_data["Ever_Married"] = label.fit_transform(test_data["Ever_Married"])
test_data["Graduated"] = label.fit_transform(test_data["Graduated"])
test_data["Profession"] = label.fit_transform(test_data["Profession"])
test_data["Spending_Score"] = label.fit_transform(test_data["Spending_Score"])
test_data["Var_1"] = label.fit_transform(test_data["Var_1"])

X_test = test_data.drop(["ID"], axis=1).values
ID = test_data["ID"]

# Y_pred = c.predict(X_test)
Y_pred = RForest.predict(X_test)

with open('pred.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    header = ['ID', 'Segmentation']
    writer.writerow(header)
    x = []
    # write a row to the csv file
    for i in range(0, len(Y_pred)):
        x.append([str(ID[i]), str(Y_pred[i])])
    writer.writerows(x)
    # close the file
    f.close()

score=RForest.score(X,Y)
print(score)