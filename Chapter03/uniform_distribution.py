# ------------------------------------------------------------
# 一様乱数の生成と分布の可視化
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. サンプル数 N = 100 の場合
# ------------------------------------------------------------
a = 1  # 乱数の下限値
b = 100  # 乱数の上限値
N = 100  # サンプル数

# [a, b) の範囲で一様分布乱数を N 個生成
X1 = np.random.uniform(a, b, N)

# 生成した乱数の値をプロット（系列として表示）
plt.plot(X1)
plt.title("Uniform Random Sequence (N=100)")
plt.xlabel("Sample Index")
plt.ylabel("Random Value")
plt.show()

# 生成した乱数のヒストグラムを描画
plt.figure()
plt.hist(X1, density=True, histtype="stepfilled", alpha=0.2)
plt.title("Histogram of Uniform Distribution (N=100)")
plt.xlabel("Value")
plt.ylabel("Frequency Density")
plt.show()

# ------------------------------------------------------------
# 2. サンプル数 N = 10000 の場合
# ------------------------------------------------------------
a = 1
b = 100
N = 10000

# より大きなサンプルで乱数生成
X2 = np.random.uniform(a, b, N)

# 乱数系列を描画（値が広範囲に散らばるため密度が高い）
plt.figure()
plt.plot(X2)
plt.title("Uniform Random Sequence (N=10000)")
plt.xlabel("Sample Index")
plt.ylabel("Random Value")
plt.show()

# ヒストグラムで分布形状を可視化
plt.figure()
plt.hist(X2, density=True, histtype="stepfilled", alpha=0.2)
plt.title("Histogram of Uniform Distribution (N=10000)")
plt.xlabel("Value")
plt.ylabel("Frequency Density")
plt.show()
