import numpy as np
from sklearn.linear_model import LinearRegression

#Datas from the water storage buildings
x = np.array([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]).reshape((-1, 1))
y = np.array([872.68, 909.32, 924.34, 964.96, 998.44, 1020.54, 1040.81, 1061.21, 1074.05, 1073.95, 1020.96])


#decleration of model
model = LinearRegression().fit(x, y)



r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")


print(f"intercept: {model.intercept_}")

print(f"slope: {model.coef_}")


y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")



x_new = np.array([2023,2024,2025,2026,2027]).reshape((-1, 1))
y_new = model.predict(x_new)

print("\n\nprediction of next 5 years total water amount as million m^3 : ")

print(y_new)