# ------------------------------------------------------------
# 勾配降下法（Gradient Descent）による 1次元最適化デモである
# 対象関数: f(x) = x^2 - 2x + 1 = (x - 1)^2
# ・解析解の最小値は x* = 1（∵ f'(x) = 2x - 2 = 0 ⇔ x = 1）である
# ・本コードは数値的に勾配降下で x* に収束する様子を確認するものである
# ・フォントは IPA 系（IPAexGothic 優先）に設定するである
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# 文字フォント設定：IPA 系を最優先にし、無ければ日本語フォントのフォールバックを用いるである
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [
    "IPAexGothic",  # 最優先（IP なんとか：IPA 系）
    "IPAGothic",
    "IPAexMincho",
    "IPAMincho",
    "Hiragino Sans",
    "Noto Sans CJK JP",
    "Meiryo",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け対策である


# ------------------------------------------------------------
# 1. 目的関数とその勾配の定義である
# ------------------------------------------------------------
def f(x: float) -> float:
    """目的関数 f(x) = (x - 1)^2（凸関数）である。"""
    return x**2 - 2 * x + 1


Gradf = lambda x: 2 * x - 2  # 勾配 f'(x) である

# ------------------------------------------------------------
# 2. 関数の形状を描画して直感を得るである
#    軸（x=0, y=0）を中央に置く体裁に整える
# ------------------------------------------------------------
x = np.linspace(-1, 3, 400)
y = f(x)

fig = plt.figure()
axdef = fig.add_subplot(1, 1, 1)
axdef.spines["left"].set_position("center")  # y軸を中央へ
axdef.spines["bottom"].set_position("zero")  # x軸を0へ
axdef.spines["right"].set_color("none")
axdef.spines["top"].set_color("none")
axdef.xaxis.set_ticks_position("bottom")
axdef.yaxis.set_ticks_position("left")
axdef.plot(x, y, "r", label="f(x) = (x - 1)^2")
axdef.scatter([1], [0], color="k", zorder=3, label="解析解 x*=1")
axdef.set_title("目的関数の形状（凸）")
axdef.legend(loc="upper right")
plt.show()

# ------------------------------------------------------------
# 3. 勾配降下法の設定である
#    ・反復式: x_{k+1} = x_k - η * f'(x_k)
#    ・この関数はL-滑らか（L=2）であるため、0 < η < 1/L (=0.5) なら収束しやすい
# ------------------------------------------------------------
ActualX = 3.0  # 初期点である
LearningRate = 0.01  # 学習率 η（小さすぎると遅く、大きすぎると発散しうる）
PrecisionValue = 1e-6  # 収束判定のしきい値（ステップ幅）である
PreviousStepSize = 1.0  # 初期ステップ幅
MaxIteration = 10000  # 反復上限
IterationCounter = 0  # 反復カウンタ

# 収束過程の可視化用履歴である
x_hist = [ActualX]
f_hist = [f(ActualX)]

# ------------------------------------------------------------
# 4. 勾配降下のメインループである
#    反復を進め、更新量がしきい値未満になれば停止する
# ------------------------------------------------------------
while PreviousStepSize > PrecisionValue and IterationCounter < MaxIteration:
    PreviousX = ActualX
    # 勾配方向へ学習率分だけ降りる（最急降下である）
    ActualX = ActualX - LearningRate * Gradf(PreviousX)
    PreviousStepSize = abs(ActualX - PreviousX)
    IterationCounter += 1

    # 途中経過を出力（ログが多い場合は適宜間引くと良い）
    print(
        "Number of iterations = ",
        IterationCounter,
        "\nActual value of x is = ",
        ActualX,
    )

    # 履歴を保存しておく
    x_hist.append(ActualX)
    f_hist.append(f(ActualX))

print("X value of f(x) minimum (numerical) = ", ActualX)

# ------------------------------------------------------------
# 5. 収束可視化：関数上に更新点を重ね、目的関数値の推移も描くである
# ------------------------------------------------------------
# (a) 関数曲線上に反復点を描画
plt.figure(figsize=(6, 4))
plt.plot(x, y, "r", label="f(x) = (x - 1)^2")
plt.scatter(
    x_hist,
    [f(xi) for xi in x_hist],
    s=12,
    c=np.linspace(0, 1, len(x_hist)),
    cmap="viridis",
    label="反復点",
)
plt.scatter([1], [0], color="k", zorder=3, label="解析解 x*=1")
plt.title("勾配降下の軌跡")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# (b) 目的関数値 f(x_k) の減少プロット
plt.figure(figsize=(6, 4))
plt.plot(f_hist, marker="o", markersize=3, linewidth=1)
plt.title("目的関数値の推移（単調減少が期待される）")
plt.xlabel("iteration k")
plt.ylabel("f(x_k)")
plt.yscale("log")  # ログ縮尺で収束の速さを見やすくする
plt.grid(alpha=0.3)
plt.show()

# ------------------------------------------------------------
# 6. 補足（理論的含意）である
#    ・本問題は強凸であり、勾配降下は適切な学習率で線形収束（幾何収束）する。
#    ・今回の更新式は x_{k+1} = (1 - 2η) x_k + 2η であり、η=0.01 のとき収縮率は 1-2η=0.98 である。
#      したがって x は指数関数的に 1 へ近づく。
#    ・解析解 x*=1 に対し、数値解 ActualX が十分近ければ収束が確認できる。
# ------------------------------------------------------------
