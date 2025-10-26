# ------------------------------------------------------------
# マルコフ連鎖による天気予測シミュレーション
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. 状態と遷移確率の定義
# ------------------------------------------------------------
np.random.seed(3)  # 再現性確保のため乱数シードを固定

# 状態（天気の種類）
StatesData = ["Sunny", "Rainy"]

# 状態遷移ラベル（単なる識別用）
TransitionStates = [["SuSu", "SuRa"], ["RaRa", "RaSu"]]

# 状態遷移確率行列（Markov Chain）
# 行：現在の状態、列：次の状態
# 例）[0.80, 0.20] → 晴れから晴れ80%、晴れから雨20%
TransitionMatrix = [[0.80, 0.20], [0.25, 0.75]]

# ------------------------------------------------------------
# 2. シミュレーション初期設定
# ------------------------------------------------------------
WeatherForecasting = []  # 各日の天気を格納
NumDays = 365  # 1年間（365日）
TodayPrediction = StatesData[0]  # 初日は晴れ（Sunny）から開始

print("Weather initial condition =", TodayPrediction)

# ------------------------------------------------------------
# 3. 天気のマルコフ過程をシミュレート
# ------------------------------------------------------------
for i in range(1, NumDays):
    # 現在の状態に基づき次の状態を確率的に決定
    if TodayPrediction == "Sunny":
        TransCondition = np.random.choice(
            TransitionStates[0], replace=True, p=TransitionMatrix[0]
        )
        if TransCondition == "SuRa":
            TodayPrediction = "Rainy"

    elif TodayPrediction == "Rainy":
        TransCondition = np.random.choice(
            TransitionStates[1], replace=True, p=TransitionMatrix[1]
        )
        if TransCondition == "RaSu":
            TodayPrediction = "Sunny"

    # 結果を記録
    WeatherForecasting.append(TodayPrediction)
    print(f"Day {i+1}: {TodayPrediction}")

# ------------------------------------------------------------
# 4. 可視化：時系列プロット
# ------------------------------------------------------------
# 状態を数値に変換して折れ線グラフで表示
numeric_weather = [1 if state == "Sunny" else 0 for state in WeatherForecasting]

plt.figure(figsize=(10, 4))
plt.plot(numeric_weather, color="orange")
plt.title("Weather Forecasting using Markov Chain (Sunny=1, Rainy=0)")
plt.xlabel("Days")
plt.ylabel("State")
plt.yticks([0, 1], ["Rainy", "Sunny"])
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 5. 可視化：ヒストグラムで天気の頻度分布を表示
# ------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(WeatherForecasting, bins=2, color="skyblue", rwidth=0.6)
plt.title("Frequency of Weather States over One Year")
plt.xlabel("Weather State")
plt.ylabel("Count")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# ------------------------------------------------------------
# 6. 理論的背景
# ------------------------------------------------------------
# このモデルは2状態マルコフ連鎖（2-State Markov Chain）に基づく。
# 状態集合 S = {Sunny, Rainy}
# 遷移行列 P =
#     [[0.80, 0.20],
#      [0.25, 0.75]]
#
# 定常分布 π は以下を満たす：
#     πP = π
#     π₁ + π₂ = 1
#
# → π = [0.56, 0.44]
# よって、長期的には約56%が晴れ、44%が雨に収束する。
