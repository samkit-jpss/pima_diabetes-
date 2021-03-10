import pandas as pd 
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
dataset = pd.read_csv("diabets.csv", header=None)
y=dataset[8]
X=dataset[[0, 1, 2, 3, 4, 5, 6, 7]]
model = Sequential()
model.add(Dense(units=12,input_dim=8,activation='relu'))
model.add(Dense(units=8,activation='relu'))
model.add(Dense(units=6,activation='relu'))
model.add(Dense(units=6,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X,y,epochs=200)
model.save("diabetes_model.h5")