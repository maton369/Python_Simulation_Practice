# ------------------------------------------------------------
# 焼きなまし法（Simulated Annealing; SA）で 1 次元関数の最小化を行うデモである
# 目的関数: f(x) = x * sin(2.1x + 1)
# ・高温（大きな温度）では悪化遷移も一定確率で受理し、大域探索を行う
# ・温度を徐々に下げるにつれて受理確率が下がり、局所探索に移行する
# ・受理確率は Metropolis ルール:  min(1, exp(-(E_new - E_cur)/T))
#   （元コードの "np.random.randn()" による受理判定は正しくは一様乱数であるため修正する）
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# （任意）グラフの日本語表記設定（ユーザー既定）
from matplotlib import rcParams

rcParams["font.family"] = "IPAexGothic"
rcParams["font.sans-serif"] = "Meiryo"

# ------------------------------------------------------------
# 1) 目的関数の定義と可視化である
# ------------------------------------------------------------
x = np.linspace(0.0, 10.0, 1000)  # 探索レンジの可視化用グリッドである


def cost_function(x: float) -> float:
    """評価関数 E(x) = x * sin(2.1x + 1) を返すである。"""
    return x * np.sin(2.1 * x + 1.0)


# 目的関数の形を確認するである
plt.figure()
plt.plot(x, cost_function(x))
plt.xlabel("x")
plt.ylabel("Cost Function E(x)")
plt.title("目的関数の可視化")
plt.grid(True, alpha=0.3)
plt.show()

# ------------------------------------------------------------
# 2) 焼きなまし法のハイパーパラメータと初期化である
# ------------------------------------------------------------
temp = 2000  # 初期温度 T_0（大きいほど悪化受理が起こりやすい）である
iter = 2000  # 反復回数（スケジューリングは t = temp/(i+1) を採用）である
step_size = 0.1  # 近傍摂動の標準偏差（探索ステップ幅）である
np.random.seed(15)  # 再現性のため乱数シード固定である

# 初期点を探索レンジ内から一様にサンプルするである
xi = np.random.uniform(x.min(), x.max())
E_xi = cost_function(xi)  # これまでの最良（best）値として保持するである

# 現在状態（受理／棄却の対象となる「鎖」の先頭）である
xit, E_xit = xi, E_xi

# 可視化用：最良値の履歴を蓄積するである
best_cost_history = [E_xi]
best_x_history = [xi]

# 直近の受理確率（ログ印字用）である
acc_prob = 1.0

# ------------------------------------------------------------
# 3) メインループ：近傍生成 → 評価 → 受理判定 → 最良更新 である
# ------------------------------------------------------------
for i in range(iter):
    # 3-1) 近傍点の生成：現在点に正規ノイズを加えるである
    xstep = xit + np.random.randn() * step_size
    E_step = cost_function(xstep)

    # 3-2) まずは「グローバルに最良」を更新するである（ hill-climbing 的判定 ）
    if E_step < E_xi:
        xi, E_xi = xstep, E_step
        best_cost_history.append(E_xi)
        best_x_history.append(xi)
        print(
            f"Iteration = {i:4d} | x_min = {xi:.6f} | Global Minimum = {E_xi:.6f} "
            f"| Acceptance Probability (prev) = {acc_prob:.3f}"
        )

    # 3-3) Metropolis 受理確率の計算である（悪化も温度に応じて受理）
    diff_energy = E_step - E_xit  # 新旧エネルギー差 ΔE = E_new - E_cur
    t = temp / (i + 1)  # 温度スケジュール T(i) = T0 / (i+1)
    acc_prob = np.exp(-diff_energy / t)  # 受理確率の原式である
    acc_prob = float(np.clip(acc_prob, 0.0, 1.0))  # 上限1にクリップするである

    # 3-4) 受理判定：
    #   ・ΔE < 0（改善）なら必ず受理
    #   ・ΔE >= 0（悪化）でも U(0,1) < acc_prob なら受理
    #   ※ 元コードは np.random.randn()（正規乱数）で判定していたが、正しくは一様乱数である
    if (diff_energy < 0.0) or (np.random.rand() < acc_prob):
        xit, E_xit = xstep, E_step

# ------------------------------------------------------------
# 4) 収束状況の可視化である（最良目的値の推移）
# ------------------------------------------------------------
plt.figure()
plt.plot(best_cost_history, "bs--", markersize=4, linewidth=1)
plt.xlabel("Improvement Step (best 更新回数)")
plt.ylabel("Best Cost E(x)")
plt.title("焼きなまし法による最良値の推移")
plt.grid(True, alpha=0.3)
plt.show()

# 最終的な最良推定値を表示するである
print(f"Estimated argmin x* = {xi:.6f},  E(x*) = {E_xi:.6f}")
