# ------------------------------------------------------------
# UAV WiFi データに対する SVC（サポートベクタ分類）と
# 特徴選択（SelectKBest + カイ二乗統計）を用いた最小実装である。
# 処理の流れ：
#   1) データ読込と概要確認
#   2) 学習/評価データ分割
#   3) 前処理なし SVC によるベースライン評価
#   4) min-max 正規化（[0,1] 射影）※chi2 が非負前提のため
#   5) chi2 による上位 k 特徴選択 → SVC 再学習・再評価
#
# 注意（重要）：
#   - 本コードは説明のための最小例であり、スケーリング/特徴選択を全データに対して fit している。
#     これはデータ漏洩（information leakage）を招く。実運用では
#       Pipeline(StandardScaler(or MinMaxScaler) → SelectKBest → SVC)
#     を用い、学習データのみで fit し、テストには transform のみを適用するべきである。
# ------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2

# ------------------------------------------------------------
# 1) データの読込と構造確認である
#    - 'UAV_WiFi.xlsx' は特徴量列と目的変数 'target' を含むと仮定する。
#    - read_excel はエンジンに openpyxl を用いる環境が多い（未導入ならエラーになる）である。
# ------------------------------------------------------------
data = pd.read_excel("UAV_WiFi.xlsx")

# DataFrame 情報（列名・型・欠損・メモリフットプリント）を確認するである
print(data.info())

# 文字列にキャストしてカテゴリ統計の概要（ユニーク数・最頻値など）を俯瞰するである
DataStatCat = data.astype("object").describe()
print(DataStatCat)

# ------------------------------------------------------------
# 2) 特徴量 X / 目的変数 Y の分割である
#    - 目的変数 'target' を除いた列を説明変数とする。
# ------------------------------------------------------------
X = data.drop("target", axis=1)
print("X shape = ", X.shape)
Y = data["target"]
print("Y shape = ", Y.shape)

# ------------------------------------------------------------
# 3) 学習/評価分割である
#    - 再現性のため random_state を固定する。
#    - クラス不均衡が懸念される場合は stratify=Y を推奨（本例では簡略化）である。
# ------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.30, random_state=1
)
print("X train shape = ", X_train.shape)
print("X test shape  = ", X_test.shape)
print("Y train shape = ", Y_train.shape)
print("Y test shape  = ", Y_test.shape)

# ------------------------------------------------------------
# 4) ベースライン SVC（前処理なし）での学習・評価である
#    - gamma='scale' は RBF カーネルのスケールを自動調整する（1 / (n_features * Var(X)))。
#    - 一般に SVC はスケーリングの影響を大きく受けるため、標準化や min-max を推奨するが、
#      まずは素のベースラインを取得するである。
# ------------------------------------------------------------
SVC_model = SVC(gamma="scale", random_state=0).fit(X_train, Y_train)
SVC_model_score = SVC_model.score(X_test, Y_test)
print(
    "Support Vector Classification Model Score (no scaling / no FS) = ", SVC_model_score
)

# ------------------------------------------------------------
# 5) 可視化（箱ひげ：先頭5特徴）である
#    - スケール差や外れ値を粗く把握するための可視化である。
# ------------------------------------------------------------
first_5_columns = X.iloc[:, 0:5]
plt.figure(figsize=(10, 5))
first_5_columns.boxplot()
# plt.show()  # 実行環境に応じて表示

# ------------------------------------------------------------
# 6) min-max スケーリングである
#    - chi2 スコアは非負特徴量を前提とするため、[0,1] に正規化するである。
#    - 重要：本例では *全データ X* に対して fit_transform しており、データ漏洩の温床になる。
#      実務では学習データで fit、学習/評価それぞれに transform を当てる構成にするである。
# ------------------------------------------------------------
X_scaled = (X - X.min()) / (X.max() - X.min())

# スケーリング後の箱ひげ（先頭5特徴）である
first_5_columns = X_scaled.iloc[:, 0:5]
plt.figure(figsize=(10, 5))
first_5_columns.boxplot()
# plt.show()

# ------------------------------------------------------------
# 7) 特徴選択（SelectKBest + chi2）である
#    - chi2 は各特徴とクラスの独立性検定に基づくスコアを算出し、上位 k 個を選ぶ。
#    - ここでも fit(X_scaled, Y) を全データに対して実施しており、情報漏洩の懸念がある。
#      厳密には学習データで fit → 学習/評価に transform を適用すべきである。
# ------------------------------------------------------------
best_input_columns = SelectKBest(chi2, k=10).fit(X_scaled, Y)
sel_index = best_input_columns.get_support()
best_X = X_scaled.loc[:, sel_index]

feature_selected = best_X.columns.values.tolist()
print("The best 10 feature selected are:", feature_selected)

# ------------------------------------------------------------
# 8) 選ばれた特徴のみで再分割→再学習・再評価である
#    - 再現性維持のため random_state を固定する。
#    - gamma='auto' は 1 / n_features を用いる（gamma='scale' との挙動差に留意）である。
# ------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    best_X, Y, test_size=0.30, random_state=1
)
print("X train shape = ", X_train.shape)
print("X test shape  = ", X_test.shape)
print("Y train shape = ", Y_train.shape)
print("Y test shape  = ", Y_test.shape)

SVC_model = SVC(gamma="auto", random_state=0).fit(X_train, Y_train)
SVC_model_score = SVC_model.score(X_test, Y_test)
print(
    "Support Vector Classification Model Score (with min-max + chi2-KBest) = ",
    SVC_model_score,
)

# ------------------------------------------------------------
# 参考（改善提案）である
#   - データ漏洩対策：Pipeline を用いて
#       Pipeline(steps=[('scaler', MinMaxScaler()),
#                       ('kbest', SelectKBest(chi2, k=10)),
#                       ('svc', SVC(gamma='scale', random_state=0))])
#     とし、train_test_split 後の学習データで fit する。
#   - 評価：StratifiedKFold の交差検証や GridSearchCV/Optuna で SVC の C と gamma を探索する。
#   - 前処理：SVC は標準化（平均0・分散1）も有効。chi2 を使わない場合は StandardScaler が定番である。
#   - 可視化：DecisionBoundaryDisplay は 2 次元で有効。高次元では PCA で 2D に落として境界を可視化する。
# ------------------------------------------------------------
