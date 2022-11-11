# /Users/matlab学习（邓桂丰）/DGF_course  # 9.1
# author: CantiLiu
# coding: UTF-8


import pandas as pd
import numpy as np
from scipy import stats

M1 = pd.read_table("data0901.txt", header=None)
M = M1.drop([0], axis=1)  # 剔除编号列

k0 = 0
k1 = 0
y0 = []
y1 = []
x0 = []
x1 = []
z0 = []
z1 = []

for i in range(len(M)):  # 按吸烟-不吸烟进行数据分类
    if M.iloc[i, 6] == 0:  # 吸烟状况，0～不吸烟；1～吸烟
        k0 += 1  # 不吸烟孕妇数量
        y0.append(M.iloc[i, 0])  # 不吸烟孕妇的新生儿体重
        x0.append(M.iloc[i, 1])  # 不吸烟孕妇的怀孕期
        z0.append(M.iloc[i, 5])  # 不吸烟孕妇的体重
    elif M.iloc[i, 6] == 1:
        k1 += 1  # 吸烟孕妇数量
        y1.append(M.iloc[i, 0])  # 吸烟孕妇的新生儿体重
        x1.append(M.iloc[i, 1])  # 吸烟孕妇的怀孕期
        z1.append(M.iloc[i, 5])  # 吸烟孕妇的体重

# 新生儿体重的点估计与区间估计
y0m = np.mean(y0)
y1m = np.mean(y1)
y0s = stats.sem(y0)
y1s = stats.sem(y1)
y0c = stats.norm.interval(0.95, y0m, y0s)
y1c = stats.norm.interval(0.95, y1m, y1s)
print("估计不吸烟孕妇新生儿体重均值：{}，标准差：{}，均值的置信区间：{}".format(y0m, y0s, y0c))
print("估计吸烟孕妇新生儿体重均值：{}，标准差：{}，均值的置信区间：{}".format(y1m, y1s, y1c))

# 对两样本均值进行假设检验 H0: y0m<=y1m  H1: y0m>y1m
yp = stats.ttest_ind(y0, y1, equal_var=True, alternative="greater")
print("不吸烟，吸烟孕妇新生儿体重两样本的均值检验：\n", yp)
print("拒绝原假设，故不吸烟孕妇新生儿体重均值>吸烟孕妇新生儿体重均值\n")
# p值小于0.05，拒绝原假设，因此不吸烟孕妇新生儿体重均值>吸烟孕妇新生儿体重均值


# 估计，检验吸烟与不吸烟孕妇新生儿体重偏低的比例
n01 = 0
n02 = 0
for i in range(k0):
    if y0[i] < 88.2:  # 统计不吸烟孕妇新生儿体重低于2500g的数量
        n01 += 1
    elif y0[i] < 999:  # 统计其余样本个数，剔除异常值
        n02 += 1
n0 = n01 + n02
q0 = n01 / n0  # 估计不吸烟孕妇新生儿体重值偏低的比例

n11 = 0
n12 = 0
for i in range(k1):
    if y1[i] < 88.2:  # 统计吸烟孕妇新生儿体重低于2500g的数量
        n11 += 1
    elif y1[i] < 999:  # 统计其余样本个数，剔除异常值
        n12 += 1
n1 = n11 + n12
q1 = n11 / n1  # 估计吸烟孕妇新生儿体重值偏低的比例
print("估计不吸烟孕妇新生儿体重值偏低的比例为：{}".format(q0))
print("估计吸烟孕妇新生儿体重值偏低的比例为：{}".format(q1))
# 对新生儿体重偏低比例的两样本q0，q1进行均值检验
qs = (q0 * (1 - q0) * (n0 - 1) + q1 * (1 - q1) * (n1 - 1)) / (n1 + n0 - 2) * (1 / n1 + 1 / n0)
qt = (q1 - q0) / np.sqrt(qs)
print("新生儿体重偏低比例的两0-1分布总体均值检验的t值为：{}\n".format(qt))

# 估计，检验吸烟与不吸烟孕妇的怀孕期
j0 = 0  # 不吸烟孕妇个数（排除孕期异常值）
xx0 = []  # 不吸烟孕妇孕期列表
for i in range(k0):
    if x0[i] < 999:  # 剔除异常值
        j0 += 1
        xx0.append(x0[i])

j1 = 0  # 吸烟孕妇个数（排除孕期异常值）
xx1 = []  # 吸烟孕妇孕期列表
for i in range(k1):
    if x1[i] < 999:
        j1 += 1
        xx1.append(x1[i])

# 不吸烟，吸烟孕妇孕期的点估计与区间估计
x0m = np.mean(xx0)
x1m = np.mean(xx1)
x0s = stats.sem(xx0)
x1s = stats.sem(xx1)
x0c = stats.norm.interval(0.95, x0m, x0s)
x1c = stats.norm.interval(0.95, x1m, x1s)
print("估计不吸烟孕妇孕期均值：{}，标准差：{}，均值的置信区间：{}".format(x0m, x0s, x0c))
print("估计吸烟孕妇孕期均值：{}，标准差：{}，均值的置信区间：{}".format(x1m, x1s, x1c))

# 对两样本均值进行假设检验 H0: x0m=x1m  H1: x0m!=x1m
xp = stats.ttest_ind(x0, x1, equal_var=True, alternative="two-sided")
print("不吸烟，吸烟孕妇孕期的均值检验：\n", xp)
print("接受原假设，故不吸烟孕妇孕期=吸烟孕妇孕期\n")

# 估计，检验不吸烟与吸烟孕妇早产（怀孕期小于37周）的比例
m01 = 0  # 不吸烟孕妇早产的数量
for i in range(j0):
    if xx0[i] < 37 * 7:
        m01 += 1
m11 = 0  # 吸烟孕妇早产的数量
for i in range(j1):
    if xx1[i] < 37 * 7:
        m11 += 1
r0 = m01/j0  # 不吸烟孕妇早产概率
r1 = m11/j1  # 吸烟孕妇早产概率
print("估计不吸烟孕妇早产比例为：{}".format(r0))
print("估计吸烟孕妇早产比例为：{}".format(r1))
# 对新生儿体重偏低比例的两样本q0，q1进行均值检验
rs = (r0 * (1 - r0) * (j0 - 1) + r1 * (1 - r1) * (j1 - 1)) / (j1 + j0 - 2) * (1 / j1 + 1 / j0)
rt = (r1 - r0) / np.sqrt(rs)
print("不吸烟，吸烟孕妇早产比例的两0-1分布总体均值检验的t值为：", rt)