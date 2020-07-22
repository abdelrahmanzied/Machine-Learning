# Import
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

# Data
BostonData = load_boston()
X = BostonData.data
y = BostonData.target

#print(X.shape)
#print(y.shape)
 

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=0, shuffle=True)

#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(X_test.shape)


# Model
LinearRegressionModel = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=-1)
LinearRegressionModel.fit(X_train, y_train)

print("Linear Regression Train Score: ",LinearRegressionModel.score(X_train, y_train ))
print("Linear Regression Test Score: ",LinearRegressionModel.score(X_test, y_test ))


# Predict
y_pred = LinearRegressionModel.predict(X_test)

#print(y_test[:5])
#print(y_pred[:5])


# Metrices
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
print('Mead Absolute Error ', MAEValue)

MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
print('Mead Squared Error ', MSEValue)

MdSEValue = median_absolute_error(y_test, y_pred, multioutput='uniform_average')
print('Mead Absolute Errorv ', MdSEValue)
