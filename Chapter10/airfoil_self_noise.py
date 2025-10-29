# ------------------------------------------------------------
# Airfoil Self-Noise（UCI）データに対する前処理・探索・回帰（LR/MLP）の最小実装である。
# 目的変数：SSP（Scaled Sound Pressure Level）
# 説明変数：Frequency, AngleAttack, ChordLength, FSVelox, SSDT
#
# 本コードでは
# 1) データ読み込みと基本統計・EDA（スケーリング・箱ひげ・相関）
# 2) 学習用の前処理（リーク防止のため学習データでフィッタ）
# 3) 線形回帰とMLP回帰の学習・評価（MSE/散布図）
# を段階的に実施するである。
# 各行に詳細な日本語コメントを付す。
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# グラフの日本語表記（ユーザー既定の設定）である
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [
    "IPAexGothic",
    "Meiryo",
    "Hiragino Sans",
    "Noto Sans CJK JP",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け防止である

# スケーリング・学習・評価で使用するscikit-learnモジュールである
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# 乱数の再現性確保である
RANDOM_STATE = 5
np.random.seed(RANDOM_STATE)

# ------------------------------------------------------------
# 1) データ読み込みと基本情報の把握である
#   ・UCI "airfoil_self_noise" は空白区切りのテキストであるため、正規表現で区切る
#   ・列名は問題文既定の6項目を付与する
# ------------------------------------------------------------
ASN_NAMES = [
    "Frequency",
    "AngleAttack",
    "ChordLength",
    "FSVelox",
    "SSDT",
    "SSP",
]  # 最後が目的変数である
DATA_PATH = "airfoil_self_noise.dat"  # 同ディレクトリ前提である

# pandasの将来非推奨回避のため sep=r"\s+" を用いるである
asn_df = pd.read_csv(DATA_PATH, sep=r"\s+", names=ASN_NAMES, engine="python")

# 先頭行・末尾行・基本情報を出力してデータ形状を確認するである
print("--- head(20) ---")
print(asn_df.head(20))
print("--- info() ---")
print(asn_df.info())
print("--- describe().T ---")
basic_stats = asn_df.describe().T
print(basic_stats)

# ------------------------------------------------------------
# 2) EDA用のスケーリング（0-1正規化）である
#    注意：ここではEDAの可視化・相関のために全データをMinMax化している。
#          学習用の前処理はこの後で「trainでfit→train/testをtransform」と分離するである。
# ------------------------------------------------------------
scaler_eda = MinMaxScaler()
asn_scaled_all = pd.DataFrame(scaler_eda.fit_transform(asn_df), columns=ASN_NAMES)

# スケーリング後の基本統計を確認するである（0-1区間を想定）
print("--- Scaled (EDA) describe().T ---")
print(asn_scaled_all.describe().T)

# 箱ひげ図で各特徴量のスケールと外れの傾向を把握するである
plt.figure(figsize=(10, 5))
asn_scaled_all.boxplot(column=ASN_NAMES)
plt.title("特徴量の箱ひげ図（Min-Maxスケーリング後；EDA）")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 相関係数（ピアソン）を算出・可視化するである
cor_mat = asn_scaled_all.corr(method="pearson")
with pd.option_context(
    "display.max_rows", None, "display.max_columns", cor_mat.shape[1]
):
    print("--- 相関行列（EDA; Min-Max後）---")
    print(cor_mat)

plt.figure(figsize=(6, 5))
plt.imshow(cor_mat, cmap="viridis", interpolation="nearest", aspect="auto")
plt.title("相関行列（EDA; Min-Max後）")
plt.colorbar()
plt.xticks(range(len(cor_mat.columns)), cor_mat.columns, rotation=30)
plt.yticks(range(len(cor_mat.columns)), cor_mat.columns)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 3) 学習データの準備（リーク防止）である
#    ・ここから先は学習用の前処理として、まず train/test に分割する
#    ・その後、学習データでフィッタしたスケーラを使って、train/test 両方を変換するである
#    ・目的変数SSPは線形回帰の解釈性のためスケールしない方針とする（必要なら別スケーラで対応）
# ------------------------------------------------------------
# 説明変数Xと目的変数yを分離するである（元のスケールで扱う）
X = asn_df.drop(columns=["SSP"]).copy()
y = asn_df["SSP"].copy()

print("X shape = ", X.shape)
print("y shape = ", y.shape)

# 学習/評価分割である（30%をテストへ）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE
)
print("X_train shape = ", X_train.shape)
print("X_test  shape = ", X_test.shape)
print("y_train shape = ", y_train.shape)
print("y_test  shape = ", y_test.shape)

# 特徴量のみMin-Maxスケールするである（trainでfit→train/testをtransform）
scaler_X = MinMaxScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_test_s = scaler_X.transform(X_test)

# ------------------------------------------------------------
# 4) 線形回帰（最小二乗）の学習・評価である
#    ・線形モデルは解釈性が高く、係数の符号・大きさで寄与方向・感度を把握できるである
# ------------------------------------------------------------
lin = LinearRegression()
lin.fit(X_train_s, y_train)
y_pred_lin = lin.predict(X_test_s)

mse_lin = mean_squared_error(y_test, y_pred_lin)
print("--- 線形回帰（OLS） ---")
print("MSE =", mse_lin)

# 係数の可読化（特徴量名と対で表示）である
coef_series = pd.Series(lin.coef_, index=X.columns, name="coef").sort_values(
    key=abs, ascending=False
)
print("線形回帰 係数（寄与の大きい順）：")
print(coef_series)

# ------------------------------------------------------------
# 5) MLPRegressor（多層パーセプトロン）の学習・評価である
#    ・非線形性を捉えるためのベースラインである
#    ・lbfgsは収束が速く安定しやすいがデータ全体を用いる最適化である
# ------------------------------------------------------------
mlp = MLPRegressor(
    hidden_layer_sizes=(50,),  # タプル明示である
    activation="relu",
    solver="lbfgs",
    tol=1e-4,
    max_iter=10_000,
    random_state=RANDOM_STATE,
)
mlp.fit(X_train_s, y_train)
y_pred_mlp = mlp.predict(X_test_s)

mse_mlp = mean_squared_error(y_test, y_pred_mlp)
print("--- MLP Regressor（ReLU, lbfgs） ---")
print("MSE =", mse_mlp)


# ------------------------------------------------------------
# 6) 予測 vs 実測の散布図である
#    ・対角線（y=x）に近いほど精度が高いことを示すである
#    ・対角線の描画範囲はテストデータと予測のmin/maxから動的に決めるである
# ------------------------------------------------------------
def plot_pred_vs_actual(y_true, y_pred, title):
    # 軸範囲の決定である
    y_min = float(np.min([y_true.min(), y_pred.min()]))
    y_max = float(np.max([y_true.max(), y_pred.max()]))
    pad = 0.02 * (y_max - y_min if y_max > y_min else 1.0)  # わずかに余白を取るである

    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.5)
    plt.plot(
        [y_min - pad, y_max + pad],
        [y_min - pad, y_max + pad],
        "r--",
        label="対角線（y=x）",
    )
    plt.xlabel("実測（SSP）")
    plt.ylabel("予測（SSP）")
    plt.title(title)
    plt.legend(loc="best")
    plt.axis("equal")
    plt.xlim(y_min - pad, y_max + pad)
    plt.ylim(y_min - pad, y_max + pad)
    plt.tight_layout()


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_pred_vs_actual(y_test, y_pred_mlp, f"MLP 回帰（MSE={mse_mlp:.3f}）")

plt.subplot(1, 2, 2)
plot_pred_vs_actual(y_test, y_pred_lin, f"線形回帰（MSE={mse_lin:.3f}）")
plt.show()

# ------------------------------------------------------------
# 7) 注意事項（実務向け補足；コード内コメントとして記す）
#   ・前処理：本コードでは特徴量のみMin-Maxスケールした。標準化（StandardScaler）も有力である。
#   ・スケーリングのリーク：EDA用スケーリングと学習用スケーリングを分離した（重要）。
#   ・評価指標：MSEの他にMAE/R^2/残差分布の確認が望ましい。
#   ・モデル選択：交差検証によるハイパーパラメータ調整（MLPの層・ユニット数等）が精度に効く。
#   ・外れ値：箱ひげ・相関だけでなく、Cookの距離やロバスト回帰の検討も有用である。
# ------------------------------------------------------------
