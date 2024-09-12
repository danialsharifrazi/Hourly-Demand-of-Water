import numpy as np 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,GRU,Dropout
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

target=24

def NetPlot(net_history):
    history=net_history.history
    losses=history['loss']
    val_losses=history['val_loss']
    plt.figure('Loss Diagram',dpi=200)
    plt.title('Loss of Training Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['Training Data','Validation Data'])      
    plt.show()



def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + n_steps
		if end_ix+target > len(sequence)-1:
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)



path='./dataset/Paper_dataset.txt'
data=np.loadtxt(path)

x=[]
y=[]
for item in data:
	x.append(item[:5])
	y.append(item[4])
	

for i in range(len(x)):
	if i+target>len(y)-1:
		break
	else:
		x[i]=x[i]
		y[i]=y[i+target]

x=np.array(x)
y=np.array(y)

m1=np.max(x)
m2=np.max(y)

x=x/np.max(x)
y=y/np.max(y)

x=x.reshape((x.shape[0],x.shape[1],1))


x_train=x[:-2000]
x_test=x[-2000:]
y_train=y[:-2000]
y_test=y[-2000:]


print('train',x_train.shape,y_train.shape)
print('test',x_test.shape,y_test.shape)
print(x_train[0])
print(y_train[0])


model = Sequential()
model.add(GRU(200, activation='relu', return_sequences=True, input_shape=(5,1)))
model.add(GRU(200, activation='relu', return_sequences=True))
model.add(GRU(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mae')
net=model.fit(x_train, y_train, epochs=20,validation_split=0.2)
NetPlot(net)


predicteds=model.predict(x_test)
actuals=y_test

predicteds=predicteds*m1
actuals=actuals*m2


MSE=mean_squared_error(actuals,predicteds)
RMSE=sqrt(mean_squared_error(actuals,predicteds))
MAE=mean_absolute_error(actuals,predicteds)

plt.figure(f'Forecasting {target} hours',dpi=200)
plt.plot(actuals,actuals,color='red')
plt.plot(actuals,predicteds,'bo',color='blue')
plt.xlabel('Observation')
plt.ylabel('Predict')
plt.title(f'Deep LSTM model for {target} hours')
plt.show()

actuals_path=f'./results/actuals_{target}_hours.txt'
predicteds_path=f'./results/predicteds_{target}_hours.txt'
metrics_path=f'./results/metrics_{target}_hours.txt'

f1=open(actuals_path,'a',)
f2=open(predicteds_path,'a')
f3=open(metrics_path,'a')

for i in range(len(predicteds)):
	st1=str(actuals[i])
	st1=st1.replace(']','')
	st1=st1.replace('[','')
	f1.write(st1+'\n\n')

	st2=str(predicteds[i])
	st2=st2.replace(']','')
	st2=st2.replace('[','')
	f2.write(st2+'\n\n')

f3.write('MSE: '+str(MSE)+'\nRMSE: '+str(RMSE)+'\nMAE: '+str(MAE))

f1.close()
f2.close()
f3.close()
