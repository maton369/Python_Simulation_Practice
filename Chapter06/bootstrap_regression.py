# ------------------------------------------------------------
# 単回帰のブートストラップ解析：係数・切片・R^2 の経験分布を推定する実験である
# 1) 疑似データ生成 → 2) 最小二乗回帰 → 3) 予測線の可視化
# 4) ブートストラップ（復元抽出）で回帰を繰り返し → 5) 係数分布の可視化と最良モデルの抽出
# ------------------------------------------------------------
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ------------------------------------------------------------
# 1. 疑似データ生成
# ------------------------------------------------------------
# 等間隔 x（0〜1）に一様雑音 U(0,1) を加えた y を生成する（切片≈0、傾き≈1 を想定）
x = np.linspace(0, 1, 100)
y = x + (np.random.rand(len(x)))  # ノイズは 0〜1 に一様である

# 外れ値や重複を模擬するため、既存サンプルから30点をランダムに追加する
for i in range(30):
    x = np.append(x, np.random.choice(x))
    y = np.append(y, np.random.choice(y))

# scikit-learn は (n_samples, n_features) 形状を要求するため 2次元へ整形する
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# ------------------------------------------------------------
# 2. 基本回帰の学習と指標
# ------------------------------------------------------------
reg_model = LinearRegression().fit(x, y)

# 決定係数 R^2（説明分散/総分散）を出力する
r_sq = reg_model.score(x, y)
print(f"R squared = {r_sq}")

# 回帰係数（傾き）と切片を取得する
alpha = float(reg_model.coef_[0])  # 傾き
print(f"slope: {reg_model.coef_}")
beta = float(reg_model.intercept_[0])  # 切片
print(f"intercept: {reg_model.intercept_}")

# 予測値を計算し、散布図と回帰直線を重ね描きして基礎的適合を確認する
y_pred = reg_model.predict(x)
plt.scatter(x, y)
plt.plot(x, y_pred, linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("OLS fit on full sample")
plt.show()

# ------------------------------------------------------------
# 3. ブートストラップによる回帰係数・切片・R^2 の経験分布推定
# ------------------------------------------------------------
boot_slopes = []  # 各ブートストラップにおける傾き
boot_interc = []  # 各ブートストラップにおける切片
r_sqs = []  # 各ブートストラップにおける R^2
n_boots = 500  # ブートストラップ反復回数（経験分布の滑らかさに影響）
num_sample = len(x)

# 元データからの復元抽出のため、結合データフレームを作成する
data = pd.DataFrame({"x": x[:, 0], "y": y[:, 0]})

# 各反復でサイズ n の標本を復元抽出し、回帰→指標を保存する
plt.figure()
for k in range(n_boots):
    sample = data.sample(n=num_sample, replace=True)  # 復元抽出（ブートストラップ標本）
    x_temp = sample["x"].values.reshape(-1, 1)
    y_temp = sample["y"].values.reshape(-1, 1)

    reg_model = LinearRegression().fit(x_temp, y_temp)
    r_sqs_temp = reg_model.score(x_temp, y_temp)
    r_sqs.append(r_sqs_temp)
    boot_interc.append(float(reg_model.intercept_[0]))
    boot_slopes.append(float(reg_model.coef_[0]))

    # 各標本での回帰直線を薄い灰色で重ね描きして不確実性の幅を可視化する
    y_pred_temp = reg_model.predict(x_temp)
    plt.plot(x_temp, y_pred_temp, color="grey", alpha=0.2)

# 元データの散布図と全標本での回帰直線も併記する
plt.scatter(x, y)
plt.plot(x, y_pred, linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Bootstrap regression lines overlay")
plt.show()

# ------------------------------------------------------------
# 4. 係数分布の可視化（経験分布）とダイアグノスティクス
# ------------------------------------------------------------
sns.histplot(data=boot_slopes, kde=True)
plt.xlabel("slope")
plt.title("Bootstrap distribution of slope")
plt.show()

sns.histplot(data=boot_interc, kde=True)
plt.xlabel("intercept")
plt.title("Bootstrap distribution of intercept")
plt.show()

# 各反復での R^2 の推移（学習標本依存の適合度変動）を可視化する
plt.plot(r_sqs)
plt.xlabel("bootstrap iteration")
plt.ylabel("R^2")
plt.title("R^2 across bootstrap samples")
plt.show()

# ------------------------------------------------------------
# 5. 最も適合（R^2 最大）のブートストラップ標本を特定し、その係数を報告する
# ------------------------------------------------------------
max_r_sq = max(r_sqs)
print(f"Max R squared = {max_r_sq}")

pos_max_r_sq = r_sqs.index(max(r_sqs))
print(f"Boot of the best Regression model = {pos_max_r_sq}")

max_slope = boot_slopes[pos_max_r_sq]
print(f"Slope of the best Regression model = {max_slope}")

max_interc = boot_interc[pos_max_r_sq]
print(f"Intercept of the best Regression model = {max_interc}")

# ------------------------------------------------------------
# 6. 補足（理論的背景の要点）
# ------------------------------------------------------------
# ・ブートストラップは母集団分布を仮定せず、観測データを母集団の代理とみなして
#   復元抽出を多数回行い、推定量の標本分布を経験的に近似する手法である。
# ・ここでは傾き・切片・R^2 の分布を推定しており、係数の不確実性や適合度の変動幅を可視化できる。
# ・必要に応じて、boot_slopes / boot_interc の百分位点（例：2.5%, 97.5%）を計算すれば
#   係数の95%信頼区間のブートストラップ推定も容易に得られる（である）。
