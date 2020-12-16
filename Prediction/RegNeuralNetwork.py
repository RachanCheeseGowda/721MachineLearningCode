##This file shows the entire code used to train and test our neural network

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

x=X
y=Y
y=np.reshape(y, (-1,1))
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(x))
xscale=scaler_x.transform(x)
print(scaler_y.fit(y))
yscale=scaler_y.transform(y)

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)


model = Sequential()
model.add(Dense(31, input_dim=31, kernel_initializer='normal', activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

history = model.fit(X_train, y_train, epochs=250, batch_size=50,  verbose=1, validation_split=0.2)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

print(history.history.keys())
plt.plot(history.history['mse'])
plt.plot(history.history['mae'])
plt.title('model performance')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['mse', 'mae'], loc='upper left')
plt.show()

Xnew = np.array([[-2.2578408616907195, -0.7619484936348196, -2.5780332793928573, -1.9008068681873072, -1.3986096481100645, -1.42322851369181, -3.6418724785872727, -3.9060410970576713, -1.7554032607606425, -1.3943150129620527, -1.3894138552044868, -1.3483989533215142, -0.8749243064663377, -0.4022884310098427, -2.6667544718571463, -1.907681185475232, -1.9270714623297287, -1.8745383118848733, -1.8770428706065658, -1.4066279663766306, -1.4222819925666383, -1.9167216073576925, -1.8937499044421475, -1.925058900708228, -1.4048608587821392, -1.3864975278692442, -1.3857455114766448, -0.925015121858419, -0.4127990174344398, -2.8373837683558767, -1.4360815700998038]])
Xnew= scaler_x.transform(Xnew)
ynew= model.predict(Xnew)
#invert normalize
ynew = scaler_y.inverse_transform(ynew)
Xnew = scaler_x.inverse_transform(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

nn_predictions =[]
for value in X:
    Xnew = np.array([value])
    Xnew= scaler_x.transform(Xnew)
    ynew= model.predict(Xnew)
    #invert normalize
    ynew = scaler_y.inverse_transform(ynew)
    Xnew = scaler_x.inverse_transform(Xnew)
    nn_predictions.append(ynew[0])
    #print(ynew[0])


team_arr = (team_season_old.team =="BOS").values
year_arr =(team_season_old.year ==int("1946")).values
count = 0
index=-1
for value in team_arr:
    if(team_arr[count]==True):
        if(year_arr[count]==True):
            index= count
    count= count+1
print(index)

print(team_season_no_wins.values[index].tolist())


#save neural network here
model.save('NNpredictionModel.h5')
Xnew = np.array([team_season_no_wins.values[index].tolist()])
Xnew= scaler_x.transform(Xnew)
ynew= model.predict(Xnew)
#invert normalize
ynew = scaler_y.inverse_transform(ynew)
Xnew = scaler_x.inverse_transform(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

















