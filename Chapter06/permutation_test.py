# ------------------------------------------------------------
# パーミュテーション検定で分類器の有意性を検証する実験である
#  - 本物の特徴量（Iris）と、目的変数と無相関なガウス雑音特徴量を比較
#  - 分類器：決定木（DecisionTreeClassifier）
#  - 指標：正解率（accuracy）
#  - 検定：permutation_test_score（帰無仮説：y と特徴量は無関係である）
# ------------------------------------------------------------
from sklearn.datasets import load_iris
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import permutation_test_score
import matplotlib.pyplot as plt
import seaborn as sns

# 乱数固定（再現性のため）
np.random.seed(0)

# ------------------------------------------------------------
# 1. データ準備
# ------------------------------------------------------------
data = load_iris()
X = data.data  # 4次元特徴（がく片長/幅、花弁長/幅）
y = data.target  # 品種ラベル（0,1,2）

# 目的変数と独立な「無相関特徴量」を作成（ガウス雑音）
X_noise = np.random.normal(size=(len(X), X.shape[1]))

# ------------------------------------------------------------
# 2. 分類器の定義
# ------------------------------------------------------------
clf = DecisionTreeClassifier(random_state=1)

# ------------------------------------------------------------
# 3. Irisデータでのパーミュテーション検定
#    戻り値: (観測スコア, 置換スコア配列, p値)
# ------------------------------------------------------------
score_iris, perm_scores_iris, pval_iris = permutation_test_score(
    clf, X, y, scoring="accuracy", n_permutations=1000, random_state=0, n_jobs=-1
)

print(f"Score (Iris) = {score_iris:.3f}")
print(f"P-value (Iris, permutation test) = {pval_iris:.6f}")

# ------------------------------------------------------------
# 4. 無相関データ（X_noise）でのパーミュテーション検定
#    帰無仮説が実質的に真である状況のベースラインを確認する
# ------------------------------------------------------------
score_noise, perm_scores_noise, pval_noise = permutation_test_score(
    clf, X_noise, y, scoring="accuracy", n_permutations=1000, random_state=0, n_jobs=-1
)

print(f"Score (Noise) = {score_noise:.3f}")
print(f"P-value (Noise, permutation test) = {pval_noise:.6f}")


# ------------------------------------------------------------
# 5. 可視化ユーティリティ：置換スコア分布 vs 観測スコア
#    注意：ヒストグラムの x 軸は「スコア」であり、p値はスコアではないため
#          縦線として p 値を描くのは不適切である（凡例やタイトルで表示する）
# ------------------------------------------------------------
def plot_permutation_result(observed_score, perm_scores, p_value, title):
    plt.figure(figsize=(7, 4))
    ax = sns.histplot(
        perm_scores, kde=True, bins=30, color="lightsteelblue", edgecolor="black"
    )
    # 観測スコア（学習データに対する精度）を赤線で表示
    plt.axvline(
        observed_score,
        linestyle="-",
        color="red",
        linewidth=2,
        label=f"Observed score = {observed_score:.3f}",
    )
    # タイトルに p 値を併記（p値は分布の右裾にある観測スコア以上の確率）
    plt.title(f"{title}\nPermutation p-value = {p_value:.6f}")
    plt.xlabel("Accuracy score")
    plt.ylabel("Frequency")
    plt.xlim(0.0, 1.0)
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.show()


# Irisデータの分布
plot_permutation_result(
    score_iris, perm_scores_iris, pval_iris, title="Iris (real features)"
)

# 無相関データの分布
plot_permutation_result(
    score_noise, perm_scores_noise, pval_noise, title="Noise (non-informative features)"
)

# ------------------------------------------------------------
# 6. 理論的メモ（である）
# ------------------------------------------------------------
# ・permutation_test_score は、目的変数 y をランダムに置換したデータに対する指標分布を作る。
#   帰無仮説 H0: 「特徴量と y は無関係である」。
# ・観測スコアが置換分布の右端に大きく外れていれば（p 値が小さければ）、H0 を棄却できる。
# ・本実験では、Iris（実特徴）では高スコアかつ p 値が極小になりやすく、
#   無相関特徴では観測スコアが置換分布の中心付近となり p 値は大きくなるのが期待される。
