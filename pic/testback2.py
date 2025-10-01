import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv

#画图
def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        #y.append((row[2]))
        y.append(float(row[2]))
        #x.append((row[1]))
        x.append(float(row[1]))
    return x, y


mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

plt.figure()

x, y = readcsv("smooth_test_back.csv")

w, q = readcsv("smooth_backpfl_back.csv")
plt.plot(x, y, 'b', label='KD3A ')
plt.plot(w, q, 'r', label='KD3A+RM ')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.ylim(0.5, 1)  # y轴的最大值
plt.xlim(0, 25)  # x轴最大值
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.title('源域 SVHN', fontsize=16)
plt.xlabel('epoch', fontsize=16)
plt.ylabel('Accuracy(%)', fontsize=16)
plt.legend(fontsize=16)
plt.show()