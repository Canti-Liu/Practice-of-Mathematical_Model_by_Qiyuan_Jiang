# /Users/matlab学习（邓桂丰）/DGF_course  # 9.1
# author: CantiLiu
# coding: UTF-8


import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

M1 = pd.read_table("data0901.txt", header=None)
M = M1.drop([0], axis=1)  # 剔除编号列

n = 0
y = []
x1 = []
x2 = []

# 孕妇新生儿体重和怀孕期、吸烟状况的多元回归模型
# 排除异常值
for i in range(len(M)):
    if M.iloc[i, 0] < 999 and M.iloc[i, 1] < 999 and M.iloc[i, 6] < 9:  # 吸烟状况，0～不吸烟；1～吸烟
        n += 1  # 孕妇数量
        y.append(M.iloc[i, 0])  # 孕妇新生儿体重
        x1.append(M.iloc[i, 1])  # 孕妇的怀孕期
        x2.append(M.iloc[i, 6])  # 孕妇的吸烟状况，0~不吸烟，1~吸烟

# 转换为列向量
y1 = np.array(y)
xn1 = np.array(x1)
xn2 = np.array(x2)
xnn1 = xn1 * xn2  # 交互项x1*x2
X1 = np.array([xn1, xn2, xnn1]).T  # 合并xn1，xn2，xnn1

# 进行线性回归
lrm1 = linear_model.LinearRegression()  # 生成模型对象
lrm1.fit(X1, y1)  # 计算模型
Y_pred1 = lrm1.predict(X1)  # 用模型做一个预测
lrm1_coef = lrm1.coef_  # 估计系数
lrm1_intercept = lrm1.intercept_  # 截距
lrm1_mse = mean_squared_error(y1, Y_pred1)  # 均方误差
lrm1_r2 = r2_score(y1, Y_pred1)  # 决定系数R^2
print("模型结果：\n估计系数b_i:{}\n截距b_0：{}\n均方误差MSE：{}\n决定系数R^2：{}\n".format(lrm1_coef, lrm1_intercept,
                                                                     lrm1_mse, lrm1_r2))

# 残差分析
r1 = y1 - Y_pred1  # 残差
r1c = []  # 残差置信区间
r1c_low = []
r1c_high = []
r1x = []
for i in range(len(r1)):
    r1ci = stats.t.interval(0.95, 1, loc=np.mean(r1[i]), scale=stats.sem(r1))
    r1c.append(r1ci)
    r1c_low.append(r1ci[0].tolist())
    r1c_high.append(r1ci[1].tolist())
    r1x.append(i)
# 列表降至1维
r1c_low = np.array(r1c_low).flatten().tolist()
r1c_high = np.array(r1c_high).flatten().tolist()
# 作图：残差的置信区间
plt.plot(r1x, r1c_low, 'b', marker='.', markersize=0.0, linewidth=0.5)
plt.plot(r1x, r1c_high, 'b', marker='.', markersize=0.0, linewidth=0.5)
plt.fill_between(r1x, r1c_low, r1c_high, color='blue', alpha=0.15)
plt.title("Conf-Interval of Residuals")
plt.show()

# 排除残差置信区间包含0的样本点
r_sum = 0
xn11 = xn1
xn21 = xn2
xnn2 = xnn1
y2 = y1
for i in range(n):
    if r1c_low[i] * r1c_high[i] > 0:
        r_sum += 1
        xn11 = np.delete(xn1, i)
        xn21 = np.delete(xn2, i)
        xnn2 = np.delete(xnn1, i)
        y2 = np.delete(y1, i)
print("排除{}个残差置信区间包含0的异常点".format(r_sum))

# 转换为列向量
y2 = np.array(y2)
xn11 = np.array(xn11)
xn21 = np.array(xn21)
xnn2 = xn11 * xn21
X2 = np.array([xn11, xn21, xnn2]).T  # 合并xn1，xn2，xnn2

# 进行线性回归
lrm2 = linear_model.LinearRegression()  # 生成模型对象
lrm2.fit(X2, y2)  # 计算模型
Y_pred2 = lrm2.predict(X2)  # 用模型做一个预测
lrm2_coef = lrm2.coef_  # 估计系数
lrm2_intercept = lrm2.intercept_  # 截距
lrm2_mse = mean_squared_error(y2, Y_pred2)  # 均方误差
lrm2_r2 = r2_score(y2, Y_pred2)  # 决定系数R^2
print("模型结果：\n估计系数b_i:{}\n截距b_0：{}\n均方误差MSE：{}\n决定系数R^2：{}\n".format(lrm2_coef, lrm2_intercept,
                                                                     lrm2_mse, lrm2_r2))
