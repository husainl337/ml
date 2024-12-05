import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

## LOAD THE IRIS DATASET
from sklearn.datasets import load_iris
x = load_iris(as_frame=True)

## MAKE THE DATA FRAME USING .FRAME FUNCTION OR (pd.DataFrame())
# print(x.frame.info())

## SHOWS THE SUMMARY OF STATISTICS OR NUMERICAL ATTRIBUTES
# print(x.frame.describe())

##SHOWS THE DESIRED STATS
# a = ["sepal length (cm)", "petal length (cm)"]
# print(x.frame[a].describe())


## SHOWS COUNTS OF VALUES, MIN, MAX OF A COLUMN
# print(x.frame["sepal length (cm)"].value_counts())
# print(x.frame["sepal length (cm)"].max())
# print(x.frame["sepal length (cm)"].min())


## HISTOGRAM FOR ALL ATTRUBUTES
x.frame.hist(figsize=(12, 10), bins = 30, edgecolor = "black")
plt.subplots_adjust(hspace=0.7, wspace=0.4)
# plt.show()

## HISTOGRAM FOR A SINGLE ATTRIBUTE
x.frame["sepal length (cm)"].hist(figsize=(12, 10), bins = 30, edgecolor = "black")
plt.subplots_adjust(hspace=0.7, wspace=0.4)
# plt.show()


## CHECK FOR MISSING VALUES AND FILLS THEM WITH MEDIAN OF EACH COLUMN
missVal = x.frame.isna().sum()
x.frame.fillna(x.frame.median(), inplace=True)

## SCALE NUMERICAL FEATURES TO MAKE IT COMPARABLE, SCALING DATA TO HAVE ZERO MEAN AND UNIT VARIANCE
from sklearn.preprocessing import StandardScaler
## X = INPUT FEATURES AND Y = TARGET VARIABLE
X = pd.DataFrame(x.data, columns=x.feature_names)
y = pd.Series(x.target, name="target")

# print(X.head())
# print(y.head())

## CALL THE SCALING
scale = StandardScaler()
X_scaled = scale.fit_transform(X)
# print(X_scaled)


## SPLIT DATA INTO TRAINING=80% AND TESTING=20%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

## INITIALIZE DECISION TREE MODEL
DecTreeModel = DecisionTreeClassifier()
## TRAIN THE MODEL FOR PERFECT FIT LINE ON TRAINING SET
DecTreeModel.fit(X_train, y_train)
## MAKE PREICTIONS ON TEST DATA
y_pred = DecTreeModel.predict(X_test)

## MAKE THE CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
## CALCULATE ACCURACY, PRECISION, RECALL, F1 SCORE
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("Confusion matrix: \n", cm)

print("\nAccuracy: ", accuracy)
print("precision (weighted): ", precision)
print("recall (weighted): ", recall)
print("f1 score (weighted): ", f1)