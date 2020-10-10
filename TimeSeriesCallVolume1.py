# Multilayer Perceptron to Predict International Airline Passengers (t+1, given t)
import numpy
import numpy as  np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def createSupervisedTrainingSet(dataset,lookback):

    df = pd.DataFrame()
    x = dataset
    
    len_series = x.shape[0]

    df['t'] = [x[i] for i in range(x.shape[0])]
    #create x values at time t
    x=df['t'].values
    
    cols=list()
  
    df['t+1'] = df['t'].shift(-lookback)
    cols.append(df['t+1'])
    df['t+2'] = df['t'].shift(-(lookback+1))
    cols.append(df['t+2'])
    df['t+3'] = df['t'].shift(-(lookback+2))
    cols.append(df['t+3'])
    agg = pd.concat(cols,axis=1)
    y=agg.values

    x = x.reshape(x.shape[0],1)

    len_X = len_series-lookback-2
    X=np.zeros((len_X,lookback,1))
    Y=np.zeros((len_X,3))
 
    for i in range(len_X):
        X[i] = x[i:i+lookback]
        Y[i] = y[i]

    return X,Y




# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
series = pd.read_csv('/home/anirban/Downloads/311callmetricsbymonth.csv', header=0)
series['month'] = pd.to_datetime(series.month)
series=series.sort_values(by='month',ascending=True)
series = series.iloc[:,0:2]
print(len(series))
dataset = series.values[:,1:2]
dataset = dataset.astype('float32')
print(len(dataset))
print(dataset[0:5])

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 3
#trainX, trainY = create_dataset(train, look_back)
#testX, testY = create_dataset(test, look_back)

trainX, trainY = createSupervisedTrainingSet(train, look_back)
testX,testY = createSupervisedTrainingSet(test, look_back)

print(trainX[0:5])
print(trainY[0:5])
testY=testY.reshape(testY.shape[0],testY.shape[1])
trainY=trainY.reshape(trainY.shape[0],trainY.shape[1])
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

print1 = trainY[75,:].reshape(1,-1)
print("Train X at index 93")
print(np.around(scaler.inverse_transform(trainX[75,:,:])))
print("Train Y at index 93")
print(np.around(scaler.inverse_transform(print1)))
print("Actual Data")
print(np.around(scaler.inverse_transform(dataset[75:88]))) 


model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(20, input_shape=(look_back, 1)))
model.add(tf.keras.layers.Dense(3))
myOptimizer = tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=myOptimizer, metrics=['mae'])
history = model.fit(trainX, trainY, epochs=200,  validation_data=(testX,testY), batch_size=32, verbose=0)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], color=  'red')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'], color=  'red')
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Once the model is trained, use it to make a prediction on the test data
testPredict = model.predict(testX)
predictUnscaled = np.around(scaler.inverse_transform(testPredict))
testYUnscaled = np.around(scaler.inverse_transform(testY))
#print the actual and predicted values at t+3
print("Actual values of Call Volume")
print(testYUnscaled[:,2])
print("Predicted values of Call Volume")
print(predictUnscaled[:,2])

pyplot.plot(testPredict[:,0])
pyplot.plot(testY[:,0],color='red')
pyplot.legend(['Actual','Predicted'])
pyplot.title('Actual vs Predicted at time t+1')
pyplot.show()

pyplot.plot(testPredict[:,1])
pyplot.plot(testY[:,1],color='red')
pyplot.legend(['Actual','Predicted'])
pyplot.title('Actual vs Predicted at time t+2')
pyplot.show()

pyplot.plot(testPredict[:,2])
pyplot.plot(testY[:,2],color='red')
pyplot.legend(['Actual','Predicted'])
pyplot.title('Actual vs Predicted at time t+3')
pyplot.show()

#Evaluate the RMSE values at t+1,t+2,t+3 to compare with other approaches, and select the best approach
def evaluate_forecasts(actuals, forecasts, n_seq):
    for i in range(n_seq):
        actual = actuals[:,i]
        predicted = forecasts[:,i]
        rmse = math.sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))
        
evaluate_forecasts(testYUnscaled, predictUnscaled,3)


