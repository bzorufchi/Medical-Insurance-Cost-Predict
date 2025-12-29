import pandas as pd
import matplotlib as pb
import numpy as n

df=pd.read_csv('./backend/data/insurance.csv', usecols=['age','sex','bmi','children','smoker',
                                                        'region','charges'])
# print(df.head)
# print(df.shape)
# print(df.tail)
# print(df.info())
# print(df.describe())
# print(df.nunique())

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# print(pd.get_option('display.max_rows'))
# print(pd.get_option('display.max_columns'))

# print(df.duplicated().sum())
df=df.drop_duplicates()
# print(df.shape)
# print(df.isna().sum())
# df.duplicated().sum()

numerical_features=['age','bmi','children','charges']
categorical_features=['sex', 'smoker', 'region']

# print(df[numerical_features].describe())

# for col in numerical_features:
#     print(f"{col}")
#     print("Mean" , df[col].mean())
#     print("Median" , df[col].median())
#     print("Mode" , df[col].mode()[0])
#     print("_" * 30)

# for col in categorical_features:
#     print(df[col].value_counts())
#     print("_" * 30)

df['smoker'].value_counts(normalize=True)*100

import matplotlib.pyplot as plt
numerical_features=['age', 'bmi', 'children', 'charges']
# plt.figure(figsize=(12,8))
# for i,col in enumerate(numerical_features,1):
#     plt.subplot(2,2,i)
#     plt.hist(df[col],bins=30)
#     plt.title(f"Histogram of {col}" )
#     plt.xlabel(col)
#     plt.ylabel('frequency')

# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(10,6))
# for i,col in enumerate(numerical_features,1):
#     plt.subplot(1, 4, i)
#     plt.boxplot(df[col], vert=True)
#     plt.title(col)

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12,4))
# for i,col in enumerate(categorical_features,1):
#     plt.subplot(1, 3, i)
#     df[col].value_counts().plot(kind='bar')
#     plt.title(f'Count of {col}')
#     plt.xlabel(col)
#     plt.ylabel('Count')

# plt.tight_layout()
# plt.show() 


# plt.figure()
# plt.scatter(df['age'],df['charges'])
# plt.xlabel('Age')
# plt.ylabel('Charges')
# plt.title('Age vs Medical Charges')
# plt.show()

# plt.figure()
# plt.scatter(df['bmi'], df['charges'])
# plt.xlabel('BMI')
# plt.ylabel('Charges')
# plt.title('BMI vs Medical Charges')
# plt.show()

# plt.figure()
# plt.scatter(df['children'], df['charges'])
# plt.xlabel('Number of Children')
# plt.ylabel('Charges')
# plt.title('Children vs Medical Charges')
# plt.show()

# for status in df['smoker'].unique():
#     subset = df[df['smoker'] == status]
#     plt.scatter(subset['age'], subset['charges'], label=status)

# plt.xlabel('Age')
# plt.ylabel('Charges')
# plt.title('Age vs Charges by Smoking Status')
# plt.legend()
# plt.show()

# for col in numerical_features:
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1
    
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
    
#     outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
#     print(f"{col}")
#     print(f"Lower bound: {lower_bound}")
#     print(f"Upper bound: {upper_bound}")
#     print(f"Number of outliers: {outliers.shape[0]}")
    # print("-" * 40)


df['log_charges']=n.log(df['charges'])    

# plt.figure(figsize=(12, 4))

# plt.subplot(1, 2, 1)
# plt.hist(df['charges'], bins=30)
# plt.title('Original Charges')

# plt.subplot(1, 2, 2)
# plt.hist(df['log_charges'], bins=30)
# plt.title('Log-Transformed Charges')

# plt.tight_layout()
# plt.show()

categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

df_encoded = pd.get_dummies(
    df,
    columns=categorical_features,drop_first=True
)
df_encoded.head()
df_encoded.columns

df_encoded['log_charges'] = n.log(df_encoded['charges'])
X = df_encoded.drop(['charges', 'log_charges'], axis=1)
y = df_encoded['log_charges']

# X.shape
# y.shape
# X.info()

# print(df_encoded.head(10))
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.3,
                                               shuffle=True,
                                               random_state=42
                                               )
# print("X_train shape:",X_train.shape)
# print("X_test shape:",X_test.shape)
# print("y_train shape:",y_train.shape)
# print("y_test shape:",y_test.shape)

# print(y_train.mean(), y_test.mean())

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

coefficients=pd.DataFrame({
    'Feature':X_train.columns,
    'Coefficient':model.coef_
})

model.intercept_

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np
mae_log=mean_absolute_error(y_test,y_pred)
rmse_log=np.sqrt(mean_squared_error(y_test,y_pred))
r2_log=r2_score(y_test,y_pred)

# print("Evaluation on log_charges:")
# print("MAE (log):", mae_log)
# print("RMSE (log):", rmse_log)
# print("R^2 (log):", r2_log)


y_pred_charges = np.exp(y_pred)
y_test_charges = np.exp(y_test)

mae = mean_absolute_error(y_test_charges, y_pred_charges)
rmse = np.sqrt(mean_squared_error(y_test_charges, y_pred_charges))
r2 = r2_score(y_test_charges, y_pred_charges)

# print("\nEvaluation on original charges scale:")
# print("MAE:", mae)
# print("RMSE:", rmse)
# print("R^2:", r2)

import matplotlib.pyplot as plt

# plt.figure()
# plt.scatter(y_test, y_pred)
# plt.xlabel("Actual log_charges")
# plt.ylabel("Predicted log_charges")
# plt.title("Predicted vs Actual (log scale)")
# plt.show()

residuals = y_test - y_pred

# plt.figure()
# plt.scatter(y_pred, residuals)
# plt.axhline(0)
# plt.xlabel("Predicted values (log_charges)")
# plt.ylabel("Residuals")
# plt.title("Residuals vs Predicted")
# plt.show()

# plt.figure()
# plt.hist(residuals, bins=30)
# plt.title("Histogram of Residuals")
# plt.xlabel("Residual")
# plt.ylabel("Frequency")
# plt.show()

import scipy.stats as stats

plt.figure()
stats.probplot(residuals, dist="norm", plot=plt)
# plt.title("Q-Q Plot of Residuals")
# plt.show()


import joblib

# joblib.dump(model, "linear_regression_insurance.pkl")
# joblib.dump(X_train.columns.tolist(), "model_features.pkl")


loaded_model = joblib.load("./backend/model/linear_regression_insurance.pkl")
model_features = joblib.load("./backend/model/model_features.pkl")
y_test_pred_loaded = loaded_model.predict(X_test)

print(y_test_pred_loaded[:5])

import numpy as np

predicted_charges = np.exp(y_test_pred_loaded[:5])
print(predicted_charges)








