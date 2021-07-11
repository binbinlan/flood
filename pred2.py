import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from pandas import DataFrame
from math import sqrt,nan
from keras.models import load_model
pd.set_option('display.width',None)


data = pd.read_excel(r'G:\flood\xqs\新建文件夹\pred2.xls',index_col=None,header=0)
data = data.values


series_data = data

cols = series_data.shape[1]-6


test_data=np.zeros((series_data.shape[0],cols))

for i in range(test_data.shape[0]):
    for j in range(test_data.shape[1]):
        test_data[i, j] = series_data[i, j+6]

print(test_data.shape)
test_x=test_data[:,:-1]
test_y=test_data[:,-1]
test_y = test_y.reshape(-1, 1)

real_value = np.zeros((series_data.shape[0], test_y.shape[1]))

for i in range(real_value.shape[0]):
    real_value[i, 0] = test_y[i, 0]

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

def norm_y(input_data):
    min_value = min(input_data[:, 0])
    max_value = max(input_data[:, 0])
    for i in range(input_data.shape[0]):
        input_data[i, 0] = (input_data[i, 0] - min_value) / (max_value - min_value)
    return input_data, max_value, min_value

norm_test_x = norm_x(test_x)
norm_test_y, max_test_y, min_test_y = norm_y(test_y)

model=load_model('../model.h5')
score=model.predict(norm_test_x,verbose=0)
print(score)


plt.figure()
#my_y_ticks = np.arange(0, 1, 1)
plt.ylim(0,1)

# plt.xlim((0, 500))
# plt.yticks(my_y_ticks)
plt.plot(real_value, c='k', label='true')
plt.plot(score, c='r', label='pred')

plt.legend()
plt.show()

pred_result = DataFrame(score)
pred_result.columns=['pred']
pred_result.to_csv(r'pred_result.csv', columns=['pred'], index=1)
excel_file='pred_result.csv'
excel_data=pd.read_csv(excel_file,index_col=0)
print(excel_data.iloc[0])

# ts1=[]
# for x in range(score.shape[0]):
#     for y in range(score.shape[1]):
#         ts1.append(x)
# print(ts1)
#
#
# time_data=np.zeros((series_data.shape[0],1))
# for i in range(time_data.shape[0]):
#     for j in range(time_data.shape[1]):
#         time_data[i, j] = series_data[i, j+2]
# time=time_data.reshape(-1,1)
#
#
# d={'time':time,'pred':score}
# pred_result=DataFrame(data=d)
# print(pred_result)
