# ------------------------------------------------------------
# 感度解析（Sensitivity Analysis） - 安定版（Result列バグ回避）
# ------------------------------------------------------------
import numpy as np
import math
import pandas as pd
from sensitivity import SensitivityAnalyzer


# ------------------------------------------------------------
# 1. 評価関数の定義
# ------------------------------------------------------------
def my_func(x_1, x_2, x_3):
    return math.log(x_1 / x_2 + x_3)


# ------------------------------------------------------------
# 2. 入力パラメータ設定
# ------------------------------------------------------------
x_1 = np.arange(10, 100, 10)
x_2 = np.arange(1, 10, 1)
x_3 = np.arange(1, 10, 1)
sa_dict = {"x_1": x_1.tolist(), "x_2": x_2.tolist(), "x_3": x_3.tolist()}

# ------------------------------------------------------------
# 3. SensitivityAnalyzer 実行
# ------------------------------------------------------------
sa_model = SensitivityAnalyzer(sa_dict, my_func)

# 結果のDataFrameを取得
df = sa_model.df

# ------------------------------------------------------------
# 4. Result列の存在確認と整備
# ------------------------------------------------------------
if "Result" not in df.columns:
    df = df.rename(columns={df.columns[-1]: "Result"})
    sa_model.df = df

print("=== Sensitivity DataFrame (確認用) ===")
print(df.head())

# ------------------------------------------------------------
# 5. 可視化プロット
# ------------------------------------------------------------
plot = sa_model.plot()

# ------------------------------------------------------------
# 6. styled_dfs() の代替：独自に整形・表示
# ------------------------------------------------------------
# 各パラメータごとに平均感度をまとめる
summary = df.groupby(["x_1", "x_2"])["Result"].mean().unstack(level=1).round(4)

# 見やすい表形式に整形
styled_df = summary.style.background_gradient(cmap="viridis")
print("\n=== Styled Sensitivity Table (Preview) ===")
print(summary)
