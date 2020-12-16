#This code is similar to the Multiple Linear Model Code, however uses a PCA function for the creation of a feature set
from sklearn.preprocessing import StandardScaler
# Standardizing the features
x = StandardScaler().fit_transform(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

import pandas
from sklearn import linear_model


X = principalDf
y = Y

linregr = linear_model.LinearRegression()
linregr.fit(X, y)

simpleLM = []

count = 0
for i in (X.values):
    logpredictedprob = linregr.predict([i])
    simpleLM.append(logpredictedprob)