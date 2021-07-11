import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from pandas import DataFrame
from math import sqrt,nan

pd.set_option('display.width',None)

data = pd.read_excel(r'G:\flood\xqs\newsheet.xls',index_col=None,header=0)
data = data.values



series_data = data
# 取测试集和测试集
train_data_rows = int(73)
#test_data_rows = series_data.shape[0]-train_data_rows
test_data_rows = series_data.shape[0]
cols = series_data.shape[1]-2


train_data = np.zeros((train_data_rows, cols))
test_data = np.zeros((test_data_rows, cols))

for i in range(train_data.shape[0]):
    for j in range(train_data.shape[1]):
        train_data[i, j] = series_data[i, j+2]

for i in range(test_data.shape[0]):
    for j in range(test_data.shape[1]):
        #test_data[i, j] = series_data[i+train_data_rows, j]
        test_data[i, j] = series_data[i , j+2]


# 分解输入与输出
train_x = train_data[:, :-1]
train_y = train_data[:, -1]
train_y = train_y.reshape(-1, 1)
test_x = test_data[:, :-1]
test_y = test_data[:, -1]
test_y = test_y.reshape(-1, 1)


real_values = np.zeros((test_y.shape[0], test_y.shape[1]))
for i in range(real_values.shape[0]):
    real_values[i, 0] = test_y[i, 0]
print(real_values)

real_value = np.zeros((series_data.shape[0], test_y.shape[1]))
for i in range(real_value.shape[0]):
    if i < train_y.shape[0]:
        real_value[i, 0] = train_y[i, 0]
    else:
        real_value[i, 0] = test_y[i, 0]#test_y[i-train_y.shape[0], 0]
print(real_value.shape)
# 定义输入的归一化方法
def norm_x(input_data):
    max_value = list()
    min_value = list()
    for j in range(input_data.shape[1]):
        max_value.append(max(input_data[:, j]))
        min_value.append(min(input_data[:, j]))
    for j in range(input_data.shape[1]):
        for i in range(input_data.shape[0]):
            if (max_value[j] - min_value[j]) != 0:
                input_data[i, j] = round((input_data[i, j] - min_value[j]) / (max_value[j] - min_value[j]), 3)
    return input_data


# 定义输出的归一化方法，此时需返回最大值与最小值以备反归一化
def norm_y(input_data):
    min_value = min(input_data[:, 0])
    max_value = max(input_data[:, 0])
    for i in range(input_data.shape[0]):
        input_data[i, 0] = (input_data[i, 0] - min_value) / (max_value - min_value)
    return input_data, max_value, min_value

# 生成归一化的序列数据
norm_train_x = norm_x(train_x)
# print('norm_train_x:',norm_train_x)
norm_test_x = norm_x(test_x)
norm_train_y, max_train_y, min_train_y = norm_y(train_y)
# print('norm_train_y:',norm_train_y)
norm_test_y, max_test_y, min_test_y = norm_y(test_y)

# 搭建模型与训练模型方法
def build_model(neurons, input_x, input_y, epochs, batch_size, learning_rate=0.0001, verbose=1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_x.shape[1], activation='tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(neurons * 2, activation='tanh'))
    model.add(Dropout(0.25))
    # model.add(Dense(neurons * 2, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)
    model.compile(loss='mse', optimizer=Adam(learning_rate))
    history = model.fit(input_x, input_y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=verbose,
                        validation_split=0.1)
    plt.plot(history.history['loss'], c='k', label='train_loss')
    plt.plot(history.history['val_loss'], c='r', label='test_loss')
    plt.legend()
    plt.show()
    return model

model_lstm = build_model(20, norm_train_x, train_y, epochs=200, batch_size=4)

pred_values = model_lstm.predict(norm_test_x)

print('shape is ',norm_test_x.shape)
pred_values = pred_values.reshape(-1, 1)

pred = np.zeros((pred_values.shape[0], pred_values.shape[1]))
print(pred.shape)
# 反归一化
for i in range(pred_values.shape[0]):
    pred[i, 0] = pred_values[i, 0]*(max_test_y-min_test_y)+min_test_y

# 定义计算误差MAPE的方法
def cal_MAPE(pred, true):
    error = 0
    for i in range(pred.shape[0]):
        error += (abs(pred[i, 0]-true[i, 0])/true[i, 0])
    error = error / pred.shape[0]
    return error
# 计算RMSE
def cal_RMSE(pred, true):
    error = 0
    for i in range(pred.shape[0]):
        error += (pred[i, 0]-true[i, 0])**2
    error = error / pred.shape[0]
    error = sqrt(error)
    return error
# 计算MAE
def cal_MAE(pred, true):
    error = 0
    for i in range(pred.shape[0]):
        error += abs(pred[i, 0]- true[i, 0])
    error = error/pred.shape[0]
    return error

real_pred = np.zeros((real_value.shape[0], pred.shape[1]))
for i in range(real_pred.shape[0]):
    if i < train_y.shape[0]:
        real_pred[i, 0] = pred[i,0]#nan
    else:
        real_pred[i, 0] = pred[i, 0]
        #real_pred[i, 0] = pred[i-train_y.shape[0], 0]

#MAPE = cal_MAPE(pred, real_values)
MAE = cal_MAE(pred, real_values)
RMSE = cal_RMSE(pred, real_values)
#print("MAPE is: "+str(MAPE*100)+'%')
print("MAE is: "+str(MAE))
print("RMSE is: "+str(RMSE))
# 作图
my_x_ticks = np.arange(0, 85, 1)
plt.figure()
plt.xticks(my_x_ticks)
plt.plot(real_value, c='k', label='true')
plt.plot(real_pred, c='r',label='pred')
plt.legend()
#plt.show()

#设置二值化
bina_pred=DataFrame(real_pred)
for i in range(real_pred.shape[0]):
    if real_pred[i] >= 0.5:
        real_pred[i] = 1
    else:
        real_pred[i] = 0

bina_real=DataFrame(real_value)
plt.figure()
my_x_ticks = np.arange(0, 60, 1)
my_y_ticks = np.arange(0, 1, 1)
plt.xticks(my_x_ticks)
plt.yticks(my_x_ticks)
plt.plot(bina_real,'.', c='k', label='true_binary')
plt.plot(bina_pred,'.', c='r',label='pred_binary')
plt.legend()
plt.show()


model_lstm.save(r'model.h5')
# 输出预测结果
pred = DataFrame(pred)
pred.to_csv(r'pred.csv', header=None, index=None)