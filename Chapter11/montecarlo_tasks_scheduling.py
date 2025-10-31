# ------------------------------------------------------------
# モンテカルロによるプロジェクト所要時間の推定デモである
# - 各タスクの所要時間は三角分布 Triangular(a, m, b) に従うと仮定する
#   a: 楽観値（最短） m: 最頻値 b: 悲観値（最長）
# - 乱数 u ~ U(0,1) に対する三角分布の逆関数サンプリングは次式で与えられる：
#     u < (m-a)/(b-a) のとき     x = a + sqrt( u (b-a)(m-a) )
#     それ以外のとき            x = b - sqrt( (1-u)(b-a)(b-m) )
# - ワークフローの従属性（DAG）は以下の直列・並列結合である：
#     Task1 → max(Task2, Task3) → max(Task4, Task5) → Task6
#   よって総所要時間 = T1 + max(T2,T3) + max(T4,T5) + T6 である
# - N 回の試行を行い、基本統計量とヒストグラムを出力する
# ------------------------------------------------------------

import pandas as pd
import random
import numpy as np

# シミュレーション試行回数（レプリケーション数）である
N = 10000

# 各試行の総所要時間を格納するリストである
TotalTime = []

# 各試行×各タスクの所要時間を格納する配列である（N 行 × 6 列）
T = np.empty(shape=(N, 6))

# 各タスクの三角分布パラメータ [a, m, b]（楽観・最頻・悲観）である
TaskTimes = [
    [3, 5, 8],  # Task1
    [2, 4, 7],  # Task2
    [3, 5, 9],  # Task3
    [4, 6, 10],  # Task4
    [3, 5, 9],  # Task5
    [2, 6, 8],  # Task6
]

# しきい値 Lh = (m-a)/(b-a) をタスクごとに前計算する（CDF のモード位置）である
Lh = []
for i in range(6):
    a, m, b = TaskTimes[i]
    Lh.append((m - a) / (b - a))

# 逆変換法により三角分布からサンプルを生成し、クリティカルパスに従って総所要時間を算出するである
for p in range(N):
    for i in range(6):
        u = (
            random.random()
        )  # 一様乱数 U(0,1)（Python 標準の random；速度重視なら NumPy でも可）
        a, m, b = TaskTimes[i]
        if u < Lh[i]:
            # 左側（a→m）区間の逆関数
            T[p][i] = a + np.sqrt(u * (b - a) * (m - a))
        else:
            # 右側（m→b）区間の逆関数
            T[p][i] = b - np.sqrt((1 - u) * (b - a) * (b - m))

    # 直列・並列結合を反映した総所要時間の合成である
    total = (
        T[p][0]  # Task1
        + np.maximum(T[p][1], T[p][2])  # max(Task2, Task3)
        + np.maximum(T[p][3], T[p][4])  # max(Task4, Task5)
        + T[p][5]  # Task6
    )
    TotalTime.append(total)

# タスク別サンプルの DataFrame 化と要約統計の表示である
Data = pd.DataFrame(T, columns=["Task1", "Task2", "Task3", "Task4", "Task5", "Task6"])
pd.set_option("display.max_columns", None)
print(Data.describe())  # 各タスクの平均・標準偏差・四分位など

# タスク別所要時間のヒストグラム（粗い分布形の確認）である
hist = Data.hist(bins=10)

# 総所要時間の基本統計である（最小・平均・最大）
print("Minimum project completion time = ", np.amin(TotalTime))
print("Mean project completion time = ", np.mean(TotalTime))
print("Maximum project completion time = ", np.amax(TotalTime))

# ------------------------------------------------------------
# 備考：
# - 乱数シードを固定して再現性を確保したい場合は、random.seed(seed) と
#   np.random.default_rng(seed)（または np.random.seed）を併用することが望ましい。
# - 三角分布は値域が [a,b] に有界で、モード m を明示できるため、PERT 的な三点見積もりに適する。
# - ただし不確実性が大きい場合や裾の重さを表現したい場合は、Beta/Lognormal などの
#   代替分布を検討するのがよいである。
# ------------------------------------------------------------
