# ------------------------------------------------------------
# モンテカルロ法による円周率 (π) の推定
# ------------------------------------------------------------
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. パラメータ設定
# ------------------------------------------------------------
N = 10000  # 試行回数（生成する乱数点の数）
M = 0  # 円の中に入った点の数（カウンタ）

# 円内・円外の座標を記録するリスト
XCircle = []  # 円の中にある点（x座標）
YCircle = []  # 円の中にある点（y座標）
XSquare = []  # 円の外にある点（x座標）
YSquare = []  # 円の外にある点（y座標）

# ------------------------------------------------------------
# 2. [0,1]×[0,1] の単位正方形内で乱数を生成し、円との位置関係を判定
# ------------------------------------------------------------
for p in range(N):
    x = random.random()  # x座標（0〜1の乱数）
    y = random.random()  # y座標（0〜1の乱数）

    # 半径1の1/4円（原点中心）内に入っているかどうかを判定
    if x**2 + y**2 <= 1:
        M += 1
        XCircle.append(x)
        YCircle.append(y)
    else:
        XSquare.append(x)
        YSquare.append(y)

# ------------------------------------------------------------
# 3. 円周率の推定
# ------------------------------------------------------------
# 単位正方形の面積：1
# 1/4円の面積：π/4
# よって π ≈ 4 × (円内点数 / 総点数)
Pi = 4 * M / N

print("N=%d M=%d Pi=%.2f" % (N, M, Pi))

# ------------------------------------------------------------
# 4. 理論曲線（円の境界）を生成
# ------------------------------------------------------------
XLin = np.linspace(0, 1, 100)
YLin = [math.sqrt(1 - x**2) for x in XLin]

# ------------------------------------------------------------
# 5. 結果をプロット
# ------------------------------------------------------------
plt.axis("equal")  # 縦横比を1:1に固定
plt.grid(which="major")
plt.plot(XLin, YLin, color="red", linewidth=4, label="Quarter Circle")
plt.scatter(XCircle, YCircle, color="yellow", marker=".", label="Inside Circle")
plt.scatter(XSquare, YSquare, color="blue", marker=".", label="Outside Circle")
plt.title("Monte Carlo Method for Pi Estimation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
