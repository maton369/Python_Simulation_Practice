# ------------------------------------------------------------
# 大数の法則と中心極限定理の可視化
# 一様分布から抽出した標本平均の分布を確認する
# ------------------------------------------------------------
import random
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. 母集団（母分布）の生成
# ------------------------------------------------------------
a = 1  # 下限
b = 100  # 上限
N = 10000  # 母集団の大きさ

# 一様分布 [a, b) から N 個の乱数を生成
DataPop = list(np.random.uniform(a, b, N))

# 母集団のヒストグラムを描画
plt.hist(DataPop, density=True, histtype="stepfilled", alpha=0.2)
plt.title("Population Distribution (Uniform)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

# ------------------------------------------------------------
# 2. 標本平均の分布（中心極限定理の確認）
# ------------------------------------------------------------
SamplesMeans = []  # 各標本の平均を格納するリスト

# 母集団から100個ずつ抽出し、その平均を1000回計算
for i in range(1000):
    DataExtracted = random.sample(DataPop, k=100)  # 標本抽出
    DataExtractedMean = np.mean(DataExtracted)  # 標本平均を計算
    SamplesMeans.append(DataExtractedMean)

# 標本平均の分布をヒストグラムで表示
plt.figure()
plt.hist(SamplesMeans, density=True, histtype="stepfilled", alpha=0.2)
plt.title("Sampling Distribution of the Mean (n=100, 1000 samples)")
plt.xlabel("Sample Mean")
plt.ylabel("Density")
plt.show()
