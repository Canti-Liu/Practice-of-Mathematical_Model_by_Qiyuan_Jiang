# /Users/matlab学习（邓桂丰）/DGF_course  # 9.1
# author: CantiLiu
# coding: UTF-8


import numpy as np
import pandas as pd
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
x3 = []
x4 = []
x5 = []
x6 = []
# 排除异常值
for i in range(len(M)):
    if M.iloc[i, 0] < 999 and M.iloc[i, 1] < 999 and M.iloc[i, 2] < 9 and M.iloc[i, 3] < 99 \
            and M.iloc[i, 4] < 99 and M.iloc[i, 5] < 999 and M.iloc[i, 6] < 9:  # 吸烟状况，0～不吸烟；1～吸烟
        n += 1  # 孕妇数量
        y.append(M.iloc[i, 0])  # 孕妇新生儿体重
        x1.append(M.iloc[i, 1])  # 孕妇的怀孕期
        x2.append(M.iloc[i, 2])  # 孕妇的胎次状况
        x3.append(M.iloc[i, 3])  # 孕妇的年龄
        x4.append(M.iloc[i, 4])  # 孕妇的身高
        x5.append(M.iloc[i, 5])  # 孕妇的体重
        x6.append(M.iloc[i, 6])  # 孕妇的吸烟状况

data = pd.DataFrame(data={
    'x1': x1,
    'x2': x2,
    'x3': x3,
    'x4': x4,
    'x5': x5,
    'x6': x6,
    'y': y, }
)
data = data.values.copy()


# 计算回归系数，参数
def get_regre_coef(X, Y):
    S_xy = 0
    S_xx = 0
    S_yy = 0
    # 计算预报因子和预报对象的均值
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    for i in range(len(X)):
        S_xy += (X[i] - X_mean) * (Y[i] - Y_mean)
        S_xx += pow(X[i] - X_mean, 2)
        S_yy += pow(Y[i] - Y_mean, 2)
    return S_xy / pow(S_xx * S_yy, 0.5)


# 构建原始增广矩阵
def get_original_matrix():
    # 创建一个数组存储相关系数,data.shape几行(维)几列，结果用一个tuple表示
    # print(data.shape[1])
    col = data.shape[1]
    # print(col)
    r = np.ones((col, col))  # np.ones参数为一个元组(tuple)
    # print(np.ones((col,col)))
    # for row in data.T:#运用数组的迭代，只能迭代行，迭代转置后的数组，结果再进行转置就相当于迭代了每一列
    # print(row.T)
    for i in range(col):
        for j in range(col):
            r[i, j] = get_regre_coef(data[:, i], data[:, j])
    return r


# 计算公差贡献与方差比
def get_vari_contri(r):
    col = data.shape[1]
    # 创建一个矩阵来存储方差贡献值
    v = np.ones((1, col - 1))
    # print(v)
    for i in range(col - 1):
        # v[0,i]=pow(r[i,col-1],2)/r[i,i]
        v[0, i] = pow(r[i, col - 1], 2) / r[i, i]
    return v


# 选择因子是否进入方程，
# 参数说明：r为增广矩阵，v为方差贡献值，k为方差贡献值最大的因子下标,p为当前进入方程的因子数
def select_factor(r, v, k, p):
    row = data.shape[0]  # 样本容量
    col = data.shape[1] - 1  # 预报因子数
    # 计算方差比
    f = (row - p - 2) * v[0, k - 1] / (r[col, col] - v[0, k - 1])
    # print(calc_vari_contri(r))
    return f


# 逐步回归分析与计算
# 通过矩阵转换公式来计算各部分增广矩阵的元素值
def convert_matrix(r, k):
    col = data.shape[1]
    k = k - 1  # 从第零行开始计数
    # 第k行的元素单不属于k列的元素
    r1 = np.ones((col, col))  # np.ones参数为一个元组(tuple)
    for i in range(col):
        for j in range(col):
            if (i == k and j != k):
                r1[i, j] = r[k, j] / r[k, k]
            elif (i != k and j != k):
                r1[i, j] = r[i, j] - r[i, k] * r[k, j] / r[k, k]
            elif (i != k and j == k):
                r1[i, j] = -r[i, k] / r[k, k]
            else:
                r1[i, j] = 1 / r[k, k]
    return r1


# 选择是否剔除因子，
# 参数说明：r为增广矩阵，v为方差贡献值，k为方差贡献值最大的因子下标,t为当前进入方程的因子数
def delete_factor(r, v, k, t):
    row = data.shape[0]  # 样本容量
    col = data.shape[1] - 1  # 预报因子数
    # 计算方差比
    f = (row - t - 1) * v[0, k - 1] / r[col, col]
    # print(calc_vari_contri(r))
    return f


# 计算第零步增广矩阵
r = get_original_matrix()
# 计算方差贡献值，判断选择变量
v = get_vari_contri(r)
print("第零步各因子方差贡献率为：", v)
# 计算方差比
# print(data.shape[0])
f = select_factor(r, v, 1, 0)  # 确定选择变量x1的F值，查表
print("变量x1的F值为：", f, "\n")  # 大于0.1显著性水平下的f临界值4.06，可以引入

# 矩阵转换，计算第一步矩阵
r = convert_matrix(r, 1)
# print(r)
# 计算第一步方差贡献值，判断选择变量
v = get_vari_contri(r)
print("第一步各因子方差贡献率为：", v)
f = select_factor(r, v, 6, 1)  # 确定选择变量x6的F值，查表
print("变量x6引入检验的F值为：", f, "\n")  # 大于0.1显著性水平下的f临界值4.54，可以引入

# 矩阵转换，计算第二步矩阵
r = convert_matrix(r, 6)
# print(r)
# 计算第二步方差贡献值,判断选择变量
v = get_vari_contri(r)
print("第二步各因子方差贡献率为：", v)
f = select_factor(r, v, 2, 2)  # 确定选择变量x2引入检验的F值，查表
print("变量x2引入检验的F值为：", f)  # 大于0.1显著性水平下的f临界值5.54，可以引入
f = delete_factor(r, v, 6, 2)  # 确定选择变量x6剔除检验的F值，查表
print("变量x6剔除检验的F值为：", f, "\n")  # 大于0.1显著性水平下的f临界值5.54，不可以剔除

# 矩阵转换，计算第三步矩阵
r = convert_matrix(r, 2)
# print(r)
# 计算第三步方差贡献值,判断选择变量
v = get_vari_contri(r)
print("第三步各因子方差贡献率为：", v)
f = select_factor(r, v, 4, 3)  # 确定选择变量x4引入检验的F值，查表
print("变量x4引入检验的F值为：", f)  # 大于0.1显著性水平下的f临界值8.53，可以引入
f = delete_factor(r, v, 2, 3)  # 确定选择变量x2剔除检验的F值，查表
print("变量x2剔除检验的F值为：", f, "\n")  # 大于0.1显著性水平下的f临界值8.53，不可以剔除

# 矩阵转换，计算第四步矩阵
r = convert_matrix(r, 4)
# print(r)
# 计算第四步方差贡献值,判断选择变量
v = get_vari_contri(r)
print("第四步各因子方差贡献率为：", v)
f = select_factor(r, v, 5, 4)  # 确定选择变量x5引入检验的F值，查表
print("变量x5引入检验的F值为：", f)  # 小于0.1显著性水平下的f临界值39.86，不可以引入
f = delete_factor(r, v, 4, 4)  # 确定选择变量x4剔除检验的F值，查表
print("变量x4剔除检验的F值为：", f, "\n")  # 大于0.1显著性水平下的f临界值39.86，不可以剔除
print("此时不可以引入变量也不可以删除变量，最优回归子集为{x1, x2, x4, x6}")

# 进行多元线性回归
# 转换为列向量
y1 = np.array(y)
xn1 = np.array(x1)
xn2 = np.array(x2)
xn4 = np.array(x4)
xn6 = np.array(x6)
X1 = np.array([xn1, xn2, xn4, xn6]).T  # 合并xn1，xn2，xn4, xn6

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
xn41 = xn4
xn61 = xn6
y2 = y1
for i in range(n):
    if r1c_low[i] * r1c_high[i] > 0:
        r_sum += 1
        xn11 = np.delete(xn1, i)
        xn21 = np.delete(xn2, i)
        xn41 = np.delete(xn4, i)
        xn61 = np.delete(xn6, i)
        y2 = np.delete(y1, i)
print("排除{}个残差置信区间包含0的异常点".format(r_sum))

# 转换为列向量
y2 = np.array(y2)
xn11 = np.array(xn11)
xn21 = np.array(xn21)
xn41 = np.array(xn41)
xn61 = np.array(xn61)
X2 = np.array([xn11, xn21, xn41, xn61]).T  # 合并xn11，xn21，xn41, xn61

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
