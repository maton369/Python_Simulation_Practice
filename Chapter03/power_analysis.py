# ------------------------------------------------------------
# 統計的検出力（Power Analysis）の可視化
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.power as ssp

# ------------------------------------------------------------
# 1. 検出力解析オブジェクトの作成
# ------------------------------------------------------------
# TTestPower() は t検定（両側）の検出力計算を行うクラス
stat_power = ssp.TTestPower()

# ------------------------------------------------------------
# 2. サンプルサイズの算出
# ------------------------------------------------------------
# solve_power() に effect_size=0.5, alpha=0.05, power=0.8 を与えると
# 必要なサンプルサイズ（nobs）が自動的に計算される
sample_size = stat_power.solve_power(
    effect_size=0.5,  # 効果量（Cohen's d）
    nobs=None,  # 未知数（求めたい値）
    alpha=0.05,  # 有意水準
    power=0.8,  # 検出力（1−β）
)
print("Sample Size: {:.2f}".format(sample_size))

# ------------------------------------------------------------
# 3. サンプル数固定で検出力を求める
# ------------------------------------------------------------
# nobs=33 のときの検出力を逆に求める
power = stat_power.solve_power(
    effect_size=0.5,  # 効果量
    nobs=33,  # サンプル数
    alpha=0.05,  # 有意水準
    power=None,  # 求めたい値（検出力）
)
print("Power = {:.2f}".format(power))

# ------------------------------------------------------------
# 4. 効果量とサンプル数の関係を可視化
# ------------------------------------------------------------
# 効果量 (0.2, 0.5, 0.8, 1.0) をそれぞれ固定して、
# サンプルサイズを変化させた場合の検出力曲線を描画
effect_sizes = np.array([0.2, 0.5, 0.8, 1.0])  # 効果量の配列
sample_sizes = np.array(range(5, 500))  # サンプルサイズの範囲

# plot_power() は検出力を自動でプロットする関数
stat_power.plot_power(
    dep_var="nobs",  # 横軸をサンプルサイズに設定
    nobs=sample_sizes,  # サンプル数の範囲
    effect_size=effect_sizes,  # 各効果量のラインを描画
)

# グラフ装飾
plt.xlabel("Sample Size (n)")
plt.ylabel("Statistical Power")
plt.title("Power Curve for t-test (α=0.05)")
plt.legend([f"d={d}" for d in effect_sizes])
plt.grid(True)
plt.show()
