# ------------------------------------------------------------
# 単一銘柄ではなく 6 銘柄（ADBE, CSCO, IBM, NVDA, MSFT, HPQ）の
# 調整終値を期間 2021-01-01〜2021-12-31 で取得し、
# 日次収益率→（年率換算した）平均と標準偏差→正規近似に基づく VaR を算出するデモである。
# ・データ取得   : pandas_datareader（Yahoo! Finance）である
# ・可視化       : 6 面サブプロットで終値推移を確認するである
# ・リスク計測   : 正規分布仮定のパラメトリック VaR（Variance–Covariance 法）である
# 重要な注意点：
#   (A) np.mean / np.std は NaN を無視しないため、NaN 処理の明示が望ましいである
#   (B) 年率化の式（平均×252、標準偏差×√252）と評価期間（1日 or 年）をそろえる必要があるである
#   (C) VaR は通常「損失の正値」で表す（下側分位の符号に注意）である
# ------------------------------------------------------------

import datetime as dt
import numpy as np
import pandas_datareader.data as wb
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1) 銘柄と期間の設定である
StockList = ["ADBE", "CSCO", "IBM", "NVDA", "MSFT", "HPQ"]
StartDay = dt.datetime(2021, 1, 1)
EndDay = dt.datetime(2021, 12, 31)

# 2) データ取得（調整終値を含むパネル形式）である
#    Yahoo 側の仕様変更で失敗する場合があるため、実務では yfinance 等の代替を検討するである
StockData = wb.DataReader(StockList, "yahoo", StartDay, EndDay)
StockClose = StockData["Adj Close"]
print(StockClose.describe())  # 各銘柄の記述統計である

# 3) 調整終値の可視化（6 面サブプロット）である
fig, axs = plt.subplots(3, 2, figsize=(20, 10))
axs[0, 0].plot(StockClose["ADBE"])
axs[0, 0].set_title("ADBE")
axs[0, 1].plot(StockClose["CSCO"])
axs[0, 1].set_title("CSCO")
axs[1, 0].plot(StockClose["IBM"])
axs[1, 0].set_title("IBM")
axs[1, 1].plot(StockClose["NVDA"])
axs[1, 1].set_title("NVDA")
axs[2, 0].plot(StockClose["MSFT"])
axs[2, 0].set_title("MSFT")
axs[2, 1].plot(StockClose["HPQ"])
axs[2, 1].set_title("HPQ")
plt.tight_layout()
plt.show()

# 4) 日次収益率の算出である
#    pct_change は先頭に NaN を生むため、統計量計算前に dropna を推奨するである
StockReturns = StockClose.pct_change()
print(StockReturns.tail(15))  # 終盤 15 日分の収益率である（NaN を含む可能性に注意）

# 5) VaR の入力パラメータである
PortvolioValue = (
    1000000000.00  # ポートフォリオ評価額である（スペルは Portfolio の方が一般的である）
)
ConfidenceValue = 0.95  # 信頼水準（95%）である

# 6) 平均・標準偏差の推定である
#    注意：np.mean/np.std は NaN を無視しない（→ NaN 混入で結果が NaN になり得る）である
#    実務では pandas のメソッド（skipna=True 既定）や dropna 後に計算する方が安全である
#    例：
#      MeanStockRet = StockReturns.mean()          # 列ごとの平均（NaN スキップ）
#      StdStockRet  = StockReturns.std(ddof=1)     # 不偏標準偏差
MeanStockRet = np.mean(StockReturns)  # 現行コードを尊重（各列＝銘柄ごとの平均）
StdStockRet = np.std(StockReturns)  # 現行コードを尊重（各列＝銘柄ごとの標準偏差）

# 7) 年率換算の一貫性に関する注意である
#    ・日次→年次の年率平均は「×252」、年率ボラは「×√252」が標準である
#    ・現行コードは「÷252」「÷√252」となっており、日次量に縮小されている点に注意である
#    ・以下は現行の式をそのまま用いる（挙動説明のため）。実務ではコメントの式を推奨するである
WorkingDays2021 = 252.0
AnnualizedMeanStockRet = (
    MeanStockRet / WorkingDays2021
)  # 現行のまま（※通常は MeanStockRet * 252）
AnnualizedStdStockRet = StdStockRet / np.sqrt(
    WorkingDays2021
)  # 現行のまま（※通常は StdStockRet * √252）

# 8) 正規近似に基づく分位点である
#    VaR は下側 α 分位（α = 1 - 信頼水準）を用いる：
#      q_α = N^{-1}(α; μ, σ)
#    ここで μ, σ は選んだ時間刻みに対応した値である（年次を使うなら年次 μ, σ）である
INPD = norm.ppf(1 - ConfidenceValue, AnnualizedMeanStockRet, AnnualizedStdStockRet)

# 9) VaR の算出である
#    現行コードは VaR = PV * q_α を返し、符号は分位そのものに依存する。
#    一般には「損失の正値」として VaR = -PV * q_α（q_α < 0 を想定）とする慣例が多いである
VaR = PortvolioValue * INPD

# 10) 丸めて銘柄ごとに表示するである
RoundVaR = np.round_(VaR, 2)
for i in range(len(StockList)):
    print("Value-at-Risk for", StockList[i], "is equal to ", RoundVaR[i])

# ------------------------------------------------------------
# 参考（実務での改善ポイント：コードは変更せずコメントで示す）
# ・欠損処理：StockReturns = StockReturns.dropna() としてから統計量を計算するである
# ・推定量   ：MeanStockRet = StockReturns.mean()，StdStockRet = StockReturns.std(ddof=1) を推奨するである
# ・年率化   ：AnnualizedMean = MeanDaily * 252，AnnualizedStd = StdDaily * sqrt(252) が標準である
# ・評価期間 ：1 日 VaR を求めるなら「日次 μ, σ」を用い、年次 VaR を求めるなら「年次 μ, σ」にそろえるである
# ・符号規約 ：報告値は正の損失額とするため VaR_report = -PV * q_α が解釈しやすいである
# ・分布適合 ：正規仮定の妥当性（裾の厚さ、自己相関、ボラティリティ・クラスタ）を検証し、
#              必要に応じて t 分布や GARCH（条件付きボラ）を導入するである
# ------------------------------------------------------------
