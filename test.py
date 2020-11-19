from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
model = LinearRegression(fit_intercept=True)
model.fit(x, y)
print(model.coef_[0], model.intercept_)