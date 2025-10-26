# ------------------------------------------------------------
# モンテカルロ法による数値積分の近似計算
# ------------------------------------------------------------
import random
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. 乱数シードの固定（再現性確保）
# ------------------------------------------------------------
random.seed(2)

# ------------------------------------------------------------
# 2. 積分対象関数の定義
# ------------------------------------------------------------
# ここでは f(x) = x^2 を [0,3] の区間で積分する。
# 理論上の真値は ∫₀³ x² dx = 9。
f = lambda x: x**2
a = 0.0  # 積分区間の下端
b = 3.0  # 積分区間の上端

# ------------------------------------------------------------
# 3. f(x) の最小値・最大値を求めて矩形領域を設定
# ------------------------------------------------------------
NumSteps = 1_000_000  # 分割数（関数値のサンプリング密度）
XIntegral = []  # f(x) 以下の点（積分対象領域内）
YIntegral = []
XRectangle = []  # f(x) より上の点（矩形領域内だが関数外）
YRectangle = []

ymin = f(a)
ymax = ymin
for i in range(NumSteps):
    x = a + (b - a) * float(i) / NumSteps
    y = f(x)
    if y < ymin:
        ymin = y
    if y > ymax:
        ymax = y

# ------------------------------------------------------------
# 4. モンテカルロ法による積分近似
# ------------------------------------------------------------
# (b-a)*(ymax-ymin) が矩形領域の面積。
# f(x) 以下にランダム点が入る確率を面積比として求める。
A = (b - a) * (ymax - ymin)  # 全矩形の面積
N = 1_000_000  # 乱数点の総数
M = 0  # f(x) 以下の点の数

for k in range(N):
    x = a + (b - a) * random.random()  # x ∈ [a,b)
    y = ymin + (ymax - ymin) * random.random()  # y ∈ [ymin,ymax)
    if y <= f(x):
        M += 1
        XIntegral.append(x)
        YIntegral.append(y)
    else:
        XRectangle.append(x)
        YRectangle.append(y)

# 面積比から数値積分を近似
NumericalIntegral = M / N * A
print("Numerical integration = {:.5f}".format(NumericalIntegral))

# ------------------------------------------------------------
# 5. 理論値との比較
# ------------------------------------------------------------
# 理論的な積分値: ∫₀³ x² dx = x³/3 |_0³ = 9
AnalyticalIntegral = (b**3 - a**3) / 3
print("Analytical integration = {:.5f}".format(AnalyticalIntegral))
print("Error = {:.5f}".format(abs(NumericalIntegral - AnalyticalIntegral)))

# ------------------------------------------------------------
# 6. グラフ描画
# ------------------------------------------------------------
XLin = np.linspace(a, b, 200)
YLin = [f(x) for x in XLin]

plt.axis([0, b, 0, f(b)])  # 軸範囲を指定
plt.plot(XLin, YLin, color="red", linewidth=4, label="f(x) = x²")
plt.scatter(XIntegral, YIntegral, color="blue", marker=".", label="Inside (y ≤ f(x))")
plt.scatter(
    XRectangle, YRectangle, color="yellow", marker=".", label="Outside (y > f(x))"
)
plt.title("Numerical Integration using Monte Carlo Method")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
