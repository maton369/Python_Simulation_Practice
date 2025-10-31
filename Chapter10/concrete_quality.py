# ------------------------------------------------------------
# コンクリート圧縮強度（CCS）回帰タスクの最小実装である。
# ・データ読込 → 記述統計 → 箱ひげ図（外れ値の概観） → Min-Max 正規化（全体像の把握用）
# ・学習/評価分割 → 全結合NN（Keras）による学習 → 予測 → R^2 による評価
# 注意：
#   - 本コードはユーザー提示の元コードを尊重し、構造やAPIは極力そのままにして説明コメントのみを付与する。
#   - 回帰タスクで metrics='accuracy' は原理的に不適切である（分類の指標）。MAE/MSE等が推奨である。
#   - スケーリングは本来、学習用データに対して fit し、検証/テストには transform のみを適用すべきである
#     （データリーク回避）。ここでは元コードの流れを崩さず、注意喚起のみコメントで明示する。
# ------------------------------------------------------------

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) データの読込である
#    - 提供Excel 'concrete_data.xlsx' をカラム名つきで読込む。
#    - features_names には 8 つの説明変数と 1 つの目的変数 'CCS' を与える。
# ------------------------------------------------------------
features_names = ["Cement", "BFS", "FLA", "Water", "SP", "CA", "FA", "Age", "CCS"]
concrete_data = pd.read_excel("concrete_data.xlsx", names=features_names)

# ------------------------------------------------------------
# 2) 記述統計の出力である
#    - count/mean/std/min/四分位/最大など基本統計量を確認する。
# ------------------------------------------------------------
summary = concrete_data.describe()
print(summary)

# ------------------------------------------------------------
# 3) 箱ひげ図（外れ値の概観）である
#    - 各特徴量のスケール差はあるが、外れ値の存在や分布の歪みを視覚的に把握できる。
# ------------------------------------------------------------
sns.set(style="ticks")
sns.boxplot(data=concrete_data)
plt.title("Raw features (boxplots): scale differs across variables")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4) Min-Max スケーリングである（0〜1に線形正規化）
#    - 注意：ここでは元コード通りデータ全体に fit している。
#      実運用では train/test 分割後に、訓練データで fit → その scaler を test に transform のみ適用するべきである。
#    - scaler.fit(...) の print は学習済みScalerオブジェクトのrepr表示であり、学術的意味は薄い（デバッグ向け）
# ------------------------------------------------------------
scaler = MinMaxScaler()
print(scaler.fit(concrete_data))
scaled_data = scaler.fit_transform(
    concrete_data
)  # 元コード準拠：再fit+transform（機能的には重複）
scaled_data = pd.DataFrame(scaled_data, columns=features_names)

# 正規化後の記述統計（0〜1レンジ内に収まっていることを確認）
summary = scaled_data.describe()
print(summary)

# 正規化後の箱ひげ図（スケール差が解消され、外れ値の相対位置を比較しやすい）
sns.boxplot(data=scaled_data)
plt.title("Min-Max scaled features (boxplots): comparable scales")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5) 入出力の分割である
#    - 入力: 最初の 8 列（説明変数） / 出力: 9 列目 'CCS'（目的変数）
#    - 元コードは「正規化済みの全データ」から切り出す。実務ではスケーラのfit/transformの順序に留意。
# ------------------------------------------------------------
input_data = pd.DataFrame(scaled_data.iloc[:, :8])
output_data = pd.DataFrame(scaled_data.iloc[:, 8])

# 学習/テスト分割（ホールドアウト）
inp_train, inp_test, out_train, out_test = train_test_split(
    input_data, output_data, test_size=0.30, random_state=1
)
print(inp_train.shape)
print(inp_test.shape)
print(out_train.shape)
print(out_test.shape)

# ------------------------------------------------------------
# 6) Keras による全結合ニューラルネット（回帰）である
#    - 隠れ層: 20 → 10 → 10（ReLU）
#    - 出力層: 1（線形）→ 回帰のための連続値出力
#    - optimizer: 'adam'（勾配ベース最適化）
#    - loss: 'mean_squared_error'（二乗誤差）
#    - metrics: ['accuracy'] は分類向けであり、回帰では解釈できないため実運用では MAE/MSE を推奨。
# ------------------------------------------------------------
model = Sequential()
model.add(Dense(20, input_dim=8, activation="relu"))  # 入力8次元→隠れ20
model.add(Dense(10, activation="relu"))  # 隠れ10
model.add(Dense(10, activation="relu"))  # 隠れ10
model.add(Dense(1, activation="linear"))  # 出力1（回帰）
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

# 学習（エポック数1000、verbose=1でログ表示）
# 実務では EarlyStopping（検証損失の悪化で打ち切り）を併用するのが望ましい。
model.fit(inp_train, out_train, epochs=1000, verbose=1)

# モデル概略の出力
model.summary()

# ------------------------------------------------------------
# 7) 予測と R^2 評価である
#    - 予測値と正解の相関・分散説明率を R^2 で測る。
#    - 今回は目的変数も [0,1] スケールである点に注意（元スケールでの評価を行う場合は逆変換が必要）。
# ------------------------------------------------------------
output_pred = model.predict(inp_test)

print("Coefficient of determination = ")
print(r2_score(out_test, output_pred))

# ------------------------------------------------------------
# 補足（理論的背景の要点）である：
# ・Min-Max スケーリング：各特徴量 x を [0,1] へ線形写像する（x_min, x_max は学習データ由来が望ましい）
#     x' = (x - x_min) / (x_max - x_min)
# ・回帰評価 R^2： 1 - SSE/SST。1 に近いほど当てはまりが良いが、外挿性能は別途検証が必要である。
# ・回帰の metrics： 'accuracy' は分類誤り率の概念であり、回帰では意味をなさない。
#   Keras の metrics には 'mae'（平均絶対誤差）や 'mse'（平均二乗誤差）を用いるとよい。
# ・データリーク：スケーラーは train に対して fit し、そのパラメータで test を transform のみする。
#   本コードは元スクリプトのまま全体 fit を行っているため、実験時は順序に注意するべきである。
# ------------------------------------------------------------
