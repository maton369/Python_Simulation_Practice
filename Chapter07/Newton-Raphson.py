# ------------------------------------------------------------
# Newton 法（導関数に対する Newton-Raphson）で f(x)=x^3-2x^2-x+2 の極小点を求めるデモである
# ・停留点は f'(x)=0 の解であり、f''(x)>0 であれば極小である
# ・ラムダ式の定義（FirstDerivative, SecondDerivative）は元コードのまま保持するである
# ・付加機能：IPA系フォント設定／収束軌跡の可視化／解析解との比較／ゼロ割防止ガード
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# 文字フォント（IP なんとか＝IPA系）を優先的に使う設定である
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [
    "IPAexGothic",
    "IPAGothic",
    "IPAexMincho",
    "IPAMincho",
    "Hiragino Sans",
    "Noto Sans CJK JP",
    "Meiryo",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け対策である

# 1) 可視化用の x グリッドと関数値 y を作成するである
x = np.linspace(0, 3, 100)
y = x**3 - 2 * x**2 - x + 2

# 2) 軸（原点）を中央に配置する体裁調整である（見やすさのため）
fig = plt.figure()
axdef = fig.add_subplot(1, 1, 1)
axdef.spines["left"].set_position("center")
axdef.spines["bottom"].set_position("zero")
axdef.spines["right"].set_color("none")
axdef.spines["top"].set_color("none")
axdef.xaxis.set_ticks_position("bottom")
axdef.yaxis.set_ticks_position("left")

# 3) 関数曲線を描画するである
plt.plot(x, y, "r", label="f(x) = x^3 - 2x^2 - x + 2")
plt.legend()
plt.title("三次関数の形状")
plt.show()

# 4) グリッド上での「近似的な最小位置」を参考表示するである（連続最適解そのものではない）
print("Value of x at the minimum of the function", x[np.argmin(y)])

# 5) 一次・二次導関数をラムダ式で定義する（元コードを保持）
FirstDerivative = lambda x: 3 * x**2 - 4 * x - 1  # f'(x)
SecondDerivative = lambda x: 6 * x - 4  # f''(x)

# 6) Newton 法の初期化である
ActualX = 3  # 初期点
PrecisionValue = 0.000001  # 収束判定（更新量のしきい値）
PreviousStepSize = 1  # 初期ステップ幅
MaxIteration = 10000  # 反復上限
IterationCounter = 0  # 反復カウンタ

# 収束過程の軌跡を保存（可視化用）である
x_hist = [ActualX]
f_hist = [ActualX**3 - 2 * ActualX**2 - ActualX + 2]

# 7) Newton 更新ループである： x_{k+1} = x_k - f'(x_k)/f''(x_k)
#    数値安定性のため f''(x) ≈ 0 のときは安全に停止するである
eps_h = 1e-12
while PreviousStepSize > PrecisionValue and IterationCounter < MaxIteration:
    PreviousX = ActualX
    denom = SecondDerivative(PreviousX)
    if abs(denom) < eps_h:
        print(f"警告: f''(x) が 0 に近い（x={PreviousX:.8f}）。更新を停止するである。")
        break
    ActualX = ActualX - FirstDerivative(PreviousX) / denom  # Newton ステップ
    PreviousStepSize = abs(ActualX - PreviousX)  # 更新量の絶対値で収束判定
    IterationCounter = IterationCounter + 1
    print(
        "Number of iterations = ",
        IterationCounter,
        "\nActual value of x  is = ",
        ActualX,
    )

    # 軌跡を記録するである
    x_hist.append(ActualX)
    f_hist.append(ActualX**3 - 2 * ActualX**2 - ActualX + 2)

# 8) 推定された極小点の x 値を出力するである
print("X value of f(x) minimum = ", ActualX)

# 9) 解析解と比較するである（f'(x)=0 → 3x^2 - 4x - 1 = 0 の解）
#     x = (2 ± √7)/3。f''(x) = 6x - 4 より、(2 - √7)/3 は極大、(2 + √7)/3 は極小である。
sqrt7 = np.sqrt(7.0)
x_max = (2 - sqrt7) / 3
x_min = (2 + sqrt7) / 3
print("\n=== 解析解（検算）===")
print(
    f"x_max = (2 - √7)/3 = {x_max:.10f}  （f''(x_max)={SecondDerivative(x_max):.6f} → 極大）"
)
print(
    f"x_min = (2 + √7)/3 = {x_min:.10f}  （f''(x_min)={SecondDerivative(x_min):.6f} → 極小）"
)
print(f"数値解との誤差 |ActualX - x_min| = {abs(ActualX - x_min):.3e}")

# 10) 収束過程の可視化である
# (a) 関数曲線上に更新点を重ねる
x_dense = np.linspace(0, 3, 600)
y_dense = x_dense**3 - 2 * x_dense**2 - x_dense + 2
plt.figure(figsize=(6, 4))
plt.plot(x_dense, y_dense, "r", label="f(x)")
plt.scatter(
    x_hist,
    [xi**3 - 2 * xi**2 - xi + 2 for xi in x_hist],
    s=14,
    c=np.linspace(0, 1, len(x_hist)),
    cmap="viridis",
    label="Newton 軌跡",
)
plt.axvline(x_min, color="k", linestyle="--", linewidth=1, label=f"x_min ≈ {x_min:.4f}")
plt.title("Newton 法による停留点探索の軌跡")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# (b) 目的関数値 f(x_k) の減少プロット（ログ軸）
plt.figure(figsize=(6, 4))
plt.plot(f_hist, marker="o", markersize=3, linewidth=1)
plt.title("目的関数値の推移（Newton 法）")
plt.xlabel("iteration k")
plt.ylabel("f(x_k)")
plt.yscale("log")
plt.grid(alpha=0.3)
plt.show()
