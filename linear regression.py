import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

## LOAD THE CALIFORNIA DATASET
from sklearn.datasets import fetch_california_housing
x = fetch_california_housing(as_frame=True)

## MAKE THE DATA FRAME USING .FRAME FUNCTION OR (pd.DataFrame())
# print(x.frame.info())

## SHOWS THE SUMMARY OF STATISTICS OR NUMERICAL ATTRIBUTES
# print(x.frame.describe())

##SHOWS THE DESIRED STATS
# a = ["HouseAge", "AveRooms"]
# print(x.frame[a].describe())

## SHOWS COUNTS OF VALUES, MIN, MAX OF A COLUMN
# print(x.frame["AveRooms"].value_counts())
# print(x.frame["AveRooms"].max())
# print(x.frame['AveRooms'].min())

## HISTOGRAM FOR ALL ATTRUBUTES
x.frame.hist(figsize= (12,10), bins = 30, edgecolor = "black")
plt.subplots_adjust(hspace=0.7, wspace=0.4)
# plt.show()

## HISTOGRAM FOR A SINGLE ATTRIBUTE
x.frame["AveRooms"].hist(figsize= (12,10), bins = 30, edgecolor = "black")
plt.subplots_adjust(hspace=0.7, wspace=0.4)
# plt.show()


## CHECK FOR MISSING VALUES AND FILLS THEM WITH MEDIAN OF EACH COLUMN
# missVal = x.frame.isna().sum()
# x.frame.fillna(x.frame.median(), inplace=True)

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

## INITIALIZE LINEAR REGRESSION MODEL
model = LinearRegression()
## TRAIN THE MODEL FOR PERFECT FIT LINE ON TRAINING SET
model.fit(X_train, y_train)
## MAKE PREICTIONS ON TEST DATA
y_pred = model.predict(X_test)

## CALCULATE MEAN SQUARED EEROR, ROOT MSE, AND R2 TO FIND PERFORMANCE OF MODEL
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared = False)
r2 = r2_score(y_test, y_pred)             # OR USE "model.score(X_test, y_test)""

print(f"mse is:{mse}")
print(f'rmse is: {rmse}')
print(f"r2 is:{r2}")

## PLOT THE ACTUAL VS PREDICTED VALUES AND FIT-LINE
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color= "red", linewidth = 2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted House Value")
plt.show()