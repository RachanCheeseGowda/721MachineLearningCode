from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
mseSLR = mean_squared_error(np.array(simpleLM), y)
mseMLR = mean_squared_error(np.array(mullinpredictions), y)
mseNN = mean_squared_error(np.array(neuralNetworkResults), y)

maeSLR = mean_absolute_error(np.array(simpleLM), y)
maeMLR = mean_absolute_error(np.array(mullinpredictions), y)
maeNN = mean_absolute_error(np.array(neuralNetworkResults), y)

print("The Mean Squared Error for the Simple Linear Regression model is "+str(mseSLR))
print("The Mean Absolute Error for the Simple Linear Regression model is "+str(maeSLR))
print("The variance for the Simple Linear Prediction Model is "+str(np.var(simpleLM)))

print("\nThe Mean Squared Error for the Multiple Linear Regression model is "+str(mseMLR))
print("The Mean Absolute Error for the Multiple Linear Regression model is "+str(maeMLR))
print("The variance for the Multiple Linear Prediction Model is "+str(np.var(mullinpredictions)))

print("\nThe Mean Squared Error for the Probabilistic Regression Neural Network is "+str(mseNN))
print("The Mean Absolute Error for the Probabilistic Regression Neural Network is "+str(maeNN))
print("The variance for the  Probabilistic Regression Neural Network is "+str(np.var(neuralNetworkResults)))