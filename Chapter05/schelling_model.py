# ------------------------------------------------------------
# Schellingの分離モデル（Schelling Segregation Model）シミュレーション
# ------------------------------------------------------------
import matplotlib.pyplot as plt
from random import random


# ------------------------------------------------------------
# 1. エージェントクラスの定義
# ------------------------------------------------------------
class SchAgent:
    def __init__(self, type):
        # エージェントのタイプ（例：0=青, 1=赤）
        self.type = type
        self.ag_location()  # ランダムな初期位置を設定

    def ag_location(self):
        # エージェントの座標を [0,1] × [0,1] 内のランダム位置に設定
        self.location = random(), random()

    def euclidean_distance(self, new):
        # 他エージェントとのユークリッド距離を計算
        eu_dist = (
            (self.location[0] - new.location[0]) ** 2
            + (self.location[1] - new.location[1]) ** 2
        ) ** 0.5
        return eu_dist

    def satisfaction(self, agents):
        # 近傍エージェントを距離順にソートし、タイプが同じ割合を計算
        eu_dist = []
        for agent in agents:
            if self != agent:
                eu_distance = self.euclidean_distance(agent)
                eu_dist.append((eu_distance, agent))
        eu_dist.sort(key=lambda x: x[0])  # 距離でソート
        neigh_agent = [agent for _, agent in eu_dist[:neigh_num]]  # 近傍n体を取得
        neigh_itself = sum(self.type == agent.type for agent in neigh_agent)
        # 満足度判定：同タイプがしきい値以上なら満足
        return neigh_itself >= neigh_threshold

    def update(self, agents):
        # 満足していない場合はランダムに新しい場所へ移動
        while not self.satisfaction(agents):
            self.ag_location()


# ------------------------------------------------------------
# 2. グリッド表示関数の定義
# ------------------------------------------------------------
def grid_plot(agents, step):
    x_A, y_A, x_B, y_B = [], [], [], []
    for agent in agents:
        x, y = agent.location
        if agent.type == 0:
            x_A.append(x)
            y_A.append(y)
        else:
            x_B.append(x)
            y_B.append(y)

    # 描画
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_A, y_A, "^", markerfacecolor="blue", markersize=8, label="Type A")
    ax.plot(x_B, y_B, "o", markerfacecolor="red", markersize=8, label="Type B")
    ax.set_title(f"Schelling Segregation Model — Step {step}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


# ------------------------------------------------------------
# 3. 初期化パラメータ
# ------------------------------------------------------------
num_agents_A = 500  # タイプAの数
num_agents_B = 500  # タイプBの数
neigh_num = 8  # 近傍数
neigh_threshold = 4  # 同タイプが満足と感じる最小数（半分以上）

# ------------------------------------------------------------
# 4. エージェント生成
# ------------------------------------------------------------
agents = [SchAgent(0) for _ in range(num_agents_A)]
agents.extend(SchAgent(1) for _ in range(num_agents_B))

# ------------------------------------------------------------
# 5. シミュレーションの実行
# ------------------------------------------------------------
step = 0
k = 0  # 満足したエージェント数

while k < (num_agents_A + num_agents_B):
    print(f"Step number = {step}")
    grid_plot(agents, step)
    step += 1
    k = 0
    for agent in agents:
        old_location = agent.location
        agent.update(agents)
        if agent.location == old_location:
            k += 1
else:
    print(
        f"All agents satisfied with {neigh_threshold / neigh_num * 100:.1f}% similar neighbors."
    )

# ------------------------------------------------------------
# 6. 理論的背景
# ------------------------------------------------------------
# Schelling分離モデルとは：
# ・社会学者 Thomas C. Schelling による「自発的な社会分離」を説明するモデル。
# ・個々のエージェントは「近所の半分以上が自分と同じなら満足」という単純なルールに従う。
# ・しかし全体としては、少しの偏好でも結果的に明確な「空間的分離」が生じる。
#
# 数学的には：
#   各エージェント i は集合 A_i（近傍）の中で
#       |{j ∈ A_i : type_j == type_i}| / |A_i| >= T
#   を満たすまで位置を移動する。
#
# 本シミュレーションでは、青（Type A）と赤（Type B）の点が
# 徐々に集団化していく様子が確認できる。
