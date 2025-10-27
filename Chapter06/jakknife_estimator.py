# ------------------------------------------------------------
# ジャックナイフ法（Jackknife Method）による統計量の推定
# ------------------------------------------------------------
import random
import statistics
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. データ生成
# ------------------------------------------------------------
random.seed(5)  # 再現性を確保
PopData = [
    10 * random.random() for _ in range(100)
]  # 一様分布[0,10)の母集団データを生成


# ------------------------------------------------------------
# 2. 変動係数（Coefficient of Variation, CV）の定義
# ------------------------------------------------------------
def CVCalc(Dat):
    # 標準偏差 ÷ 平均値
    return statistics.stdev(Dat) / statistics.mean(Dat)


# 母集団データに対するCVの計算
CVPopData = CVCalc(PopData)
print(f"母集団の変動係数（CV）= {CVPopData:.4f}")

# ------------------------------------------------------------
# 3. ジャックナイフ法の準備
# ------------------------------------------------------------
N = len(PopData)
JackVal = [0] * (N - 1)  # i番目を除いたサンプル
PseudoVal = [0] * N  # 擬似値（Pseudo-values）

# ------------------------------------------------------------
# 4. ジャックナイフ推定の実装
# ------------------------------------------------------------
for i in range(N):
    # i番目のデータを除いたサブサンプルを作成
    for j in range(N):
        if j < i:
            JackVal[j] = PopData[j]
        elif j > i:
            JackVal[j - 1] = PopData[j]
    # 擬似値を計算： θ_i* = N*θ - (N-1)*θ_(i)
    PseudoVal[i] = N * CVCalc(PopData) - (N - 1) * CVCalc(JackVal)

# ------------------------------------------------------------
# 5. 擬似値の分布を可視化
# ------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.hist(PseudoVal, bins=15, color="skyblue", edgecolor="black")
plt.title("Jackknife Pseudo-values Distribution")
plt.xlabel("Pseudo-values")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 6. ジャックナイフ統計量の算出
# ------------------------------------------------------------
MeanPseudoVal = statistics.mean(PseudoVal)
VariancePseudoVal = statistics.variance(PseudoVal)
VarJack = VariancePseudoVal / N

print(f"ジャックナイフ擬似値の平均 = {MeanPseudoVal:.4f}")
print(f"ジャックナイフ擬似値の分散 = {VariancePseudoVal:.4f}")
print(f"ジャックナイフ分散推定量 VarJack = {VarJack:.6f}")

# ------------------------------------------------------------
# 7. 理論的背景
# ------------------------------------------------------------
# 【ジャックナイフ法の概要】
# ・統計量 θ（例：平均・分散・変動係数など）の分散やバイアスを推定する再標本化手法。
# ・各データ点を1つずつ除外したサブサンプル（Jackknife sample）を作り、
#   その統計量 θ_(i) から擬似値 θ_i* を計算して分散を推定する。
#
# 擬似値の式：
#     θ_i* = N*θ - (N - 1)*θ_(i)
#
# 分散推定量：
#     VarJack = Var(θ_i*) / N
#
# この手法により、ブートストラップよりも軽量に安定した分散推定が可能となる。
