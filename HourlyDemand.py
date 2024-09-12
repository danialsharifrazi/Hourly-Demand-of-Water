import numpy as np 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,Dense
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

block=24

def NetPlot(net_history):
    history=net_history.history
    losses=history['loss']
    val_losses=history['val_loss']
    plt.figure('Loss Diagram')
    plt.title('Loss of Training Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['Training Data','Validation Data'])      
    plt.show()



def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		if out_end_ix > len(sequences):
			break
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


path='D:/Developer/Python Projects/Hourly Demand/New/dataset/Paper_dataset.txt'
data=np.loadtxt(path)

in_seq1=[]
in_seq2=[]
in_seq3=[]
in_seq4=[]
out_seq=[]
for item in data:
	in_seq1.append(item[0])
	in_seq2.append(item[1])
	in_seq3.append(item[2])
	in_seq4.append(item[3])
	out_seq.append(item[4])

in_seq1=np.array(in_seq1)
in_seq2=np.array(in_seq2)
in_seq3=np.array(in_seq3)
in_seq4=np.array(in_seq4)
out_seq=np.array(out_seq)

in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq4 = in_seq4.reshape((len(in_seq4), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

dataset = np.hstack((in_seq1, in_seq2, in_seq3, in_seq4, out_seq))

n_steps_in,n_steps_out=100,block

x, y = split_sequences(dataset, n_steps_in, n_steps_out)

x_train=x[:-2000]
x_test=x[-2000:]
y_train=y[:-2000]
y_test=y[-2000:]

lst_x=[]
lst_y=[]
for i in range(0,2000,block):
	lst_x.append(x_test[i])
	lst_y.append(y_test[i])

x_test=np.array(lst_x)
y_test=np.array(lst_y)

n_features = x.shape[2]


model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))

model.compile(optimizer='adam', loss='mse')
net=model.fit(x_train, y_train, epochs=20,validation_split=0.2)
NetPlot(net)


predicteds=model.predict(x_test)
actuals=y_test


MSE=mean_squared_error(actuals,predicteds)
RMSE=sqrt(mean_squared_error(actuals,predicteds))
MAE=mean_absolute_error(actuals,predicteds)

plt.figure(f'Forecasting {block} hours')
plt.plot(actuals,actuals,color='red')
plt.plot(actuals,predicteds,'bo',color='blue')
plt.xlabel('Observation')
plt.ylabel('Predict')
plt.title(f'Deep LSTM model for {block} hours')
plt.show()

actuals_path=f'./results/actuals_{block}_hours_AllFeatures.txt'
predicteds_path=f'./results/predicteds_{block}_hours_AllFeatures.txt'
metrics_path=f'./results/metrics_{block}_hours_AllFeatures.txt'

f1=open(actuals_path,'a')
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
