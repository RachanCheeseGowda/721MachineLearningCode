#A simple algorithm, that does not use dimensionality reduction
from sklearn import linear_model


X = X
y = Y

lm = linear_model.LinearRegression()
linmodel = lm.fit(X,y)

mullinpredictions = linmodel.predict(X)
print(mullinpredictions)