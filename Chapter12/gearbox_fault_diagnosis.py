# ------------------------------------------------------------
# 欠陥検知データの多クラス分類（ロジスティック回帰 / ランダムフォレスト /
# 多層パーセプトロン / k-NN）と、2特徴に限定した決定境界の可視化である。
# 目的：
#   - 記述統計でデータの分布と型を把握する
#   - 学習/評価データに分割して汎化性能（accuracy）を比較する
#   - 2 次元（先頭2特徴）に射影したときの決定境界を可視化する
# 注意：
#   - ロジスティック回帰や MLP はスケーリングの影響が大きい。必要に応じて
#     StandardScaler 等を導入することを推奨する（本コードでは素の特徴量のままである）。
#   - DecisionBoundaryDisplay は「可視化時の X が2特徴であること」を仮定する。
#     学習済みモデルが多次元特徴を使っている場合、可視化用に「2特徴のみで再学習した
#     モデル」を別途用意するのが安全である（ここでは学習済みモデルに対し、先頭2特徴の
#     グリッドを与える実装だが、バージョンによっては不整合になる可能性がある）。
# ------------------------------------------------------------

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay

# ------------------------------------------------------------
# 1) データ読み込み
#    - 目的変数 'state'、説明変数 a1, a2, ... を含む Excel を読み込むである。
#    - 列名はファイル内に保存されている前提である（names を指定しない）。
# ------------------------------------------------------------
data = pd.read_excel("fault.dataset.xlsx")

# 先頭確認（欠損や型の当たりを付ける）
print(data.head(10))
print(data.info())

# 数値列の記述統計（件数/平均/分散/最小・最大・分位点）
DataStat = data.describe()
print(DataStat)

# 文字列（カテゴリ）としての記述統計（ユニーク数・最頻値など）を確認するである
DataStatCat = data.astype("object").describe()
print(DataStatCat)

# ------------------------------------------------------------
# 2) クラスごとの分布を箱ひげで俯瞰（ここでは a1, a2 のみ例示）
#    - 外れ値・クラス分離の効き具合の目視確認である。
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 10))
sns.boxplot(ax=axes[0], x="state", y="a1", data=data)
sns.boxplot(ax=axes[1], x="state", y="a2", data=data)
# 目盛の範囲はデータにより適宜調整すること
plt.ylim(-40, 40)
plt.show()

# ------------------------------------------------------------
# 3) 特徴量/目的変数の分割
# ------------------------------------------------------------
X = data.drop("state", axis=1)  # 説明変数
print("X shape = ", X.shape)
Y = data["state"]  # 目的変数（クラスラベル）
print("Y shape = ", Y.shape)

# 学習/評価に分割（再現性のため random_state を固定）
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.30,
    random_state=1,
    stratify=Y,  # クラス不均衡なら stratify を推奨である
)
print("X train shape = ", X_train.shape)
print("X test shape  = ", X_test.shape)
print("Y train shape = ", Y_train.shape)
print("Y test shape  = ", Y_test.shape)

# ------------------------------------------------------------
# 4) 各モデルの学習と評価（.score は分類では accuracy を返す）
#    - ロジスティック回帰：線形分離の基準線。収束警告が出る場合は max_iter を増やす。
# ------------------------------------------------------------
lr_model = LogisticRegression(random_state=0, max_iter=200).fit(X_train, Y_train)
lr_model_score = lr_model.score(X_test, Y_test)
print("Logistic Regression Model Score = ", lr_model_score)

# 決定境界の可視化（先頭2特徴で射影）
# ※ 学習は多次元で行っているため、環境によっては from_estimator が 2 次元前提と不整合になる。
#   可視化専用に X_train.iloc[:, :2] で再学習した軽量モデルを用いるのがより安全である。
ax1 = DecisionBoundaryDisplay.from_estimator(
    lr_model,  # 学習済み推定器
    X_train.iloc[:, :2],  # 可視化に使う2特徴（射影）
    response_method="predict",
    alpha=0.5,
)
ax1.ax_.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=Y_train, edgecolor="k")
ax1.ax_.set_title("Logistic Regression (2D projection)")
plt.show()

# ------------------------------------------------------------
# ランダムフォレスト：非線形・相互作用を表現できるアンサンブルである。
# max_depth は浅く抑え、過学習を回避するデモ設定とする。
# ------------------------------------------------------------
rm_model = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, Y_train)
rm_model_score = rm_model.score(X_test, Y_test)
print("Random Forest Model Score = ", rm_model_score)

ax2 = DecisionBoundaryDisplay.from_estimator(
    rm_model, X_train.iloc[:, :2], response_method="predict", alpha=0.5
)
ax2.ax_.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=Y_train, edgecolor="k")
ax2.ax_.set_title("Random Forest (2D projection)")
plt.show()

# ------------------------------------------------------------
# 多層パーセプトロン（MLP）：非線形決定境界を学習可能。
# スケーリングが無い場合、学習が不安定になることがある（要スケール検討）。
# ------------------------------------------------------------
mlp_model = MLPClassifier(random_state=1, max_iter=300).fit(X_train, Y_train)
mlp_model_score = mlp_model.score(X_test, Y_test)
print("Artificial Neural Network Model Score = ", mlp_model_score)

ax3 = DecisionBoundaryDisplay.from_estimator(
    mlp_model, X_train.iloc[:, :2], response_method="predict", alpha=0.5
)
ax3.ax_.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=Y_train, edgecolor="k")
ax3.ax_.set_title("MLP (2D projection)")
plt.show()

# ------------------------------------------------------------
# k-NN：局所多数決によるノンパラメトリック法。n_neighbors は交差検証で調整すること。
# ------------------------------------------------------------
kn_model = KNeighborsClassifier(n_neighbors=2).fit(X_train, Y_train)
kn_model_score = kn_model.score(X_test, Y_test)
print("K-nearest neighbors Model Score =", kn_model_score)

ax4 = DecisionBoundaryDisplay.from_estimator(
    kn_model, X_train.iloc[:, :2], response_method="predict", alpha=0.5
)
ax4.ax_.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=Y_train, edgecolor="k")
ax4.ax_.set_title("k-NN (2D projection)")
plt.show()

# ------------------------------------------------------------
# 参考：実務での拡張ポイント
#  - 前処理：欠損/外れ値処理、標準化/正規化、カテゴリのエンコード
#  - 評価：混同行列、F1、ROC-AUC、層別K-Fold交差検証
#  - 重要度：RF の feature_importances_、LR の係数、Permutation Importance
#  - 可視化：PCA/UMAP による次元削減プロット、SHAP/Partial Dependence など
# ------------------------------------------------------------
