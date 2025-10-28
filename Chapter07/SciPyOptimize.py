# ------------------------------------------------------------
# Matyas 関数の可視化と、2 つの導関数不要最適化法（Nelder–Mead / Powell）
# による最小化デモである。
# ・Matyas 関数: f(x, y) = 0.26(x^2 + y^2) - 0.48xy
#   凸関数であり、一意の大域最小は (x, y) = (0, 0)、最小値は 0 である。
# ・最適化: SciPy の minimize を用い、勾配を使わない単純形法（Nelder–Mead）
#   と Powell 法で解を求める。どちらも連続最適化の古典的手法である。
# ・可視化: 3D サーフェスで関数地形を描画し、関数の形状を確認するである。
# ------------------------------------------------------------

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  # 明示 import（古い API 互換のため）

# （任意）グラフの日本語表記設定（ユーザー既定）
from matplotlib import rcParams

rcParams["font.family"] = "IPAexGothic"
rcParams["font.sans-serif"] = "Meiryo"

# ------------------------------------------------------------
# 目的関数（Matyas 関数）の定義である。
# 引数 x は長さ 2 のベクトル [x0, x1] を想定する。
# 戻り値はスカラー（関数値）である。
# ------------------------------------------------------------
def matyas(x):
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


# ------------------------------------------------------------
# 可視化のためのグリッド生成である。
# x, y の一様グリッドを meshgrid で 2 次元配列に展開し、
# 各格子点における関数値 z = f(x, y) を計算する。
# ------------------------------------------------------------
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
x, y = np.meshgrid(x, y)
z = matyas([x, y])  # ベクトル化計算：配列同士の演算で全点の f 値を一括算出するである

# ------------------------------------------------------------
# 3D サーフェスの描画設定である。
# Matplotlib のバージョン差により `gca(projection=...)` が使えない環境があるため、
# 互換性の高い `fig.add_subplot(111, projection='3d')` を用いるである。
# ------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# サーフェス描画。rstride/cstride は格子間引き（古い引数名）である。
surf = ax.plot_surface(
    x,
    y,
    z,
    rstride=1,
    cstride=1,
    cmap=cm.RdBu,
    linewidth=0,
    antialiased=False,
)

# z 軸の目盛り（ロケータとフォーマッタ）を設定して見やすくするである。
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

# カラーバーで関数値のスケールを表示するである。
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.title("Matyas 関数の 3D サーフェス（最小は原点）")
plt.show()

# ------------------------------------------------------------
# SciPy による最小化（Nelder–Mead）である。
# ・Nelder–Mead：単純形（シンプレックス）を用いる導関数不要最適化。
# ・options:
#   - xatol: 解の許容誤差（単純形の大きさがこの値以下で収束判定）
#   - disp : 収束ログを表示するである。
# 初期値 x0 は [-10, 10] とする。
# ------------------------------------------------------------
x0 = np.array([-10, 10])
NelderMeadOptimizeResults = minimize(
    matyas,
    x0,
    method="nelder-mead",
    options={"xatol": 1e-8, "disp": True},
)

print("Nelder–Mead の推定解 =", NelderMeadOptimizeResults.x)

# ------------------------------------------------------------
# SciPy による最小化（Powell 法）である。
# ・Powell 法：方向セット探索に基づく導関数不要最適化。
# ・options:
#   - xtol: 変数の許容誤差（更新幅がこの値以下で収束判定）
#   - disp: 収束ログを表示するである。
# 同じ初期値 x0 から開始し、手法の差異による収束の違いを観察するである。
# ------------------------------------------------------------
x0 = np.array([-10, 10])
PowellOptimizeResults = minimize(
    matyas,
    x0,
    method="Powell",
    options={"xtol": 1e-8, "disp": True},
)

print("Powell 法の推定解   =", PowellOptimizeResults.x)

# ------------------------------------------------------------
# 参考：
# Matyas 関数は凸（正定値二次形式）であるため、どの初期値からでも
# （数値的に適切な設定なら）大域最小 (0, 0) に収束するはずである。
# ただし実務ではスケーリング（変数の桁整合）や停止条件の調整、
# ノイズのある関数ではロバスト化などが重要である。
# ------------------------------------------------------------
