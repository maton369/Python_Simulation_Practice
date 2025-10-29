# ------------------------------------------------------------
# 目的：AMZN 終値から対数収益を推定し、幾何ブラウン運動（Geometric Brownian Motion; GBM）
#       に基づくモンテカルロで将来（もしくは任意長の）価格パスをシミュレートするデモである。
# 手順：
#   1) CSV（AMZN.csv）の読み込み（Date をインデックス、Close 列のみ使用）
#   2) 可視化（実測の価格系列）
#   3) 日次リターン → 対数収益（log-returns）の計算
#   4) 対数収益の平均・分散・標準偏差からドリフト項 Drift を算出（μ - 0.5σ^2）
#   5) 正規乱数を用いたブラウン運動増分の生成（標準正規）
#   6) GBM：S_{t+1} = S_t * exp( Drift + σ * Z_t ) によりパスを構築
#   7) 実測系列と重ね書きして挙動を比較
# 備考：
#   - 本コードは最小例として「定数ドリフト・定数ボラティリティ」の GBM を仮定しているである。
#   - 推定量は単純な標本統計（平均・分散）であり、年率換算やボラティリティクラスタリング等は考慮しないである。
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from pandas.plotting import register_matplotlib_converters

# 日付型の変換に関する Pandas→Matplotlib の互換ヘルパ（古い環境向け）である
register_matplotlib_converters()

# ------------------------------------------------------------
# 1) データ読み込み：Date を DatetimeIndex、Close 列のみを使用するである
#    parse_dates=True と index_col='Date' により日時をインデックス化するである
# ------------------------------------------------------------
AmznData = pd.read_csv(
    "AMZN.csv",
    header=0,
    usecols=["Date", "Close"],
    parse_dates=True,
    index_col="Date",
)

# データの概要確認である（列型、欠損、統計量など）
print(AmznData.info())
print(AmznData.head())
print(AmznData.tail())
print(AmznData.describe())

# ------------------------------------------------------------
# 2) 実測価格系列の可視化である
# ------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(AmznData, label="AMZN Close")
plt.title("AMZN Close (observed)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 3) 日次リターン → 対数収益の計算である
#    pct_change() は (P_t / P_{t-1} - 1) を返す。
#    対数収益 r_t = log(1 + pct_change) とするである。
#    先頭行は NaN になるため、統計計算は自動的に NaN を無視する（pandas の既定）である。
# ------------------------------------------------------------
AmznDataPctChange = AmznData.pct_change()
AmznLogReturns = np.log(1 + AmznDataPctChange)  # DataFrame（1 列）を維持

print(AmznLogReturns.tail(10))

plt.figure(figsize=(10, 5))
plt.plot(AmznLogReturns, label="Log-returns")
plt.title("AMZN Log-returns")
plt.xlabel("Date")
plt.ylabel("Log-return")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4) 対数収益の統計量（μ, Var, σ）と Drift の算出である
#    GBM の離散化における 1 ステップの期待対数収益は
#      Drift = μ - (1/2) * σ^2
#    と表されるである。
#    ここでは「日次」の μ, σ をそのまま用いる最小例である（年率化は行わない）。
# ------------------------------------------------------------
MeanLogReturns = np.array(AmznLogReturns.mean())  # 形は (1,) 相当の配列になる
VarLogReturns = np.array(
    AmznLogReturns.var()
)  # 分散（不偏ではなく母分散ベース、pandas 既定）
StdevLogReturns = np.array(AmznLogReturns.std())  # 標準偏差

Drift = MeanLogReturns - (0.5 * VarLogReturns)
print("Drift = ", Drift)

# ------------------------------------------------------------
# 5) シミュレーション長（行：時間、列：シナリオ数）を設定するである
#    NumIntervals：時間ステップ数
#    Iterations  ：シナリオ本数
# ------------------------------------------------------------
NumIntervals = 2515  # 例：観測系列長に合わせたステップ数（必要に応じて調整）
Iterations = 20  # 例：20 本のモンテカルロパス

# 乱数の固定で再現性を確保するである
np.random.seed(7)

# ------------------------------------------------------------
# 6) 標準正規乱数 Z_t を生成するである
#    ここでは一様乱数→正規分位（norm.ppf）で生成しているが、
#    np.random.standard_normal((NumIntervals, Iterations)) でもよいである。
# ------------------------------------------------------------
SBMotion = norm.ppf(np.random.rand(NumIntervals, Iterations))

# ------------------------------------------------------------
# 7) GBM の 1 ステップ日次リターンを構成するである
#    S_{t+1} = S_t * exp( Drift + σ * Z_t )
#    従って日次の乗率（gross return）は exp( Drift + σ * Z_t ) になるである。
# ------------------------------------------------------------
DailyReturns = np.exp(Drift + StdevLogReturns * SBMotion)

# ------------------------------------------------------------
# 8) 初期価格の設定と価格パス配列の初期化である
#    AmznData.iloc[0] は Series（1 要素）になるため、スカラーに明示変換しておくと安全である。
# ------------------------------------------------------------
# 元コードに忠実：Series をそのまま代入（ブロードキャストに依存）
StartStockPrices = AmznData.iloc[0]
# 推奨（安全）：スカラー化
# StartPriceScalar = float(AmznData.iloc[0, 0])

StockPrice = np.zeros_like(DailyReturns)  # 形：(NumIntervals, Iterations)
StockPrice[0] = StartStockPrices  # 全シナリオ同一の初期価格で開始するである
# StockPrice[0] = StartPriceScalar        # （安全版）

# ------------------------------------------------------------
# 9) 価格パスの生成ループである（逐次乗算）
#    ベクトル化されているため、各時点の全シナリオを一括更新できるである。
# ------------------------------------------------------------
for t in range(1, NumIntervals):
    StockPrice[t] = StockPrice[t - 1] * DailyReturns[t]

# ------------------------------------------------------------
# 10) 結果の可視化である
#     - シミュレーションした複数パス（薄色線）
#     - 実測の AMZN 終値（黒星）を重ねて傾向を比較するである
# ------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(StockPrice, alpha=0.8)  # モンテカルロの複数経路
AMZNTrend = np.array(AmznData.iloc[:, 0:1])  # 実測系列（Nx1）→ numpy 配列
plt.plot(AMZNTrend, "k*", label="Observed AMZN")
plt.title("GBM Monte Carlo Paths vs AMZN Close")
plt.xlabel("Time step")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 補足メモ：
#  - 本推定では μ, σ を全期間一定と仮定しているため、現実のボラティリティクラスタリングや
#    ドリフト変動は表現されないである。実務ではロール推定や GARCH 系、年率換算（×√252）等を
#    併用するである。
#  - NumIntervals を観測長とは独立に設定すれば「将来パス」の予測としても扱えるである。
#  - SBMotion の生成は標準正規の直接生成（standard_normal）に置き換えても同等である。
# ------------------------------------------------------------
