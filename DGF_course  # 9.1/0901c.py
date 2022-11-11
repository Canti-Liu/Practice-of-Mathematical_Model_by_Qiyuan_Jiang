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

k0 = 0
k1 = 0
y0 = []
y1 = []
x0 = []
x1 = []

# 按吸烟-不吸烟进行数据分类
for i in range(len(M)):
    if M.iloc[i, 6] == 0:  # 吸烟状况，0～不吸烟；1～吸烟
        k0 += 1  # 不吸烟孕妇数量
        y0.append(M.iloc[i, 0])  # 不吸烟孕妇的新生儿体重
        x0.append(M.iloc[i, 1])  # 不吸烟孕妇的怀孕期
    elif M.iloc[i, 6] == 1:
        k1 += 1  # 吸烟孕妇数量
        y1.append(M.iloc[i, 0])  # 吸烟孕妇的新生儿体重
        x1.append(M.iloc[i, 1])  # 吸烟孕妇的怀孕期

# 不吸烟孕妇的新生儿体重和怀孕期的回归模型
n1 = 0  # 不吸烟孕妇个数（剔除异常值）
yn1 = []  # 不吸烟孕妇新生儿体重列表
xn1 = []  # 不吸烟孕妇怀孕期列表
for i in range(k0):
    if y0[i] < 999 and x0[i] < 999:  # 吸烟孕妇的新生儿体重和怀孕期数据不缺失
        n1 += 1
        yn1.append(y0[i])
        xn1.append(x0[i])
# 转换为列向量
yn1 = np.array(yn1).reshape(-1, 1)
xn1 = np.array(xn1).reshape(-1, 1)

# 进行线性回归
lrm1 = linear_model.LinearRegression()  # 生成模型对象
lrm1.fit(xn1, yn1)  # 计算模型
Y_pred = lrm1.predict(xn1)  # 用模型做一个预测
lrm1_coef = lrm1.coef_  # 估计系数
lrm1_intercept = lrm1.intercept_  # 截距
lrm1_mse = mean_squared_error(yn1, Y_pred)  # 均方误差
lrm1_r2 = r2_score(yn1, Y_pred)  # 决定系数R^2
print("模型结果：\n估计系数b_i:{}\n截距b_0：{}\n均方误差MSE：{}\n决定系数R^2：{}\n".format(lrm1_coef, lrm1_intercept,
                                                                     lrm1_mse, lrm1_r2))
plt.scatter(xn1, yn1, color="black")
plt.plot(xn1, Y_pred, color="blue", linewidth=3)
plt.show()

# 残差分析
r1 = yn1 - Y_pred  # 残差
r1c = []  # 残差置信区间
r1c_low = []
r1c_high = []
r1x = []
r1cii = []
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
xn2 = xn1
yn2 = yn1
for i in range(n1):
    if r1c_low[i] * r1c_high[i] > 0:
        r_sum += 1
        xn2 = np.delete(xn1, i)
        yn2 = np.delete(yn1, i)
print("排除{}个残差置信区间包含0的异常点".format(r_sum))
# 转换为列向量
yn2 = np.array(yn2).reshape(-1, 1)
xn2 = np.array(xn2).reshape(-1, 1)

# 进行线性回归
lrm2 = linear_model.LinearRegression()  # 生成模型对象
lrm2.fit(xn2, yn2)  # 计算模型
Y_pred2 = lrm2.predict(xn2)  # 用模型做一个预测
lrm2_coef = lrm2.coef_  # 估计系数
lrm2_intercept = lrm2.intercept_  # 截距
lrm2_mse = mean_squared_error(yn2, Y_pred2)  # 均方误差
lrm2_r2 = r2_score(yn2, Y_pred2)  # 决定系数R^2
print("模型结果：\n估计系数b_i:{}\n截距b_0：{}\n均方误差MSE：{}\n决定系数R^2：{}\n".format(lrm2_coef, lrm2_intercept,
                                                                     lrm2_mse, lrm2_r2))
plt.scatter(xn2, yn2, color="black")
plt.plot(xn2, Y_pred2, color="blue", linewidth=3)
plt.show()

# 残差分析
r2 = yn2 - Y_pred2  # 残差
r2c = []  # 残差置信区间
r2c_low = []
r2c_high = []
r2x = []
for i in range(len(r2)):
    r2ci = stats.t.interval(0.95, 1, loc=np.mean(r2[i]), scale=stats.sem(r2))
    r2c.append(r2ci)
    r2c_low.append(r2ci[0].tolist())
    r2c_high.append(r2ci[1].tolist())
    r2x.append(i)
# 列表降至1维
r2c_low = np.array(r2c_low).flatten().tolist()
r2c_high = np.array(r2c_high).flatten().tolist()
# 作图——残差的置信区间
plt.plot(r2x, r2c_low, 'b', marker='.', markersize=0.0, linewidth=0.5)
plt.plot(r2x, r2c_high, 'b', marker='.', markersize=0.0, linewidth=0.5)
plt.fill_between(r2x, r2c_low, r2c_high, color='blue', alpha=0.15)
plt.title("Conf-Interval of Residuals")
plt.show()
