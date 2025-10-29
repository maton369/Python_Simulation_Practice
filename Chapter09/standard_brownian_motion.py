# -*- coding: utf-8 -*-
"""
離散時間の標準ブラウン運動（Wiener過程）を擬似生成する最小コードである。
理論の対応関係：
  ・連続時間 T=1 を n 分割して Δt = 1/n とする。
  ・ブラウン運動の独立増分は  N(0, Δt)  に従うため，
      ΔB_k = sqrt(Δt) * Z_k = (1/√n) * Z_k,  Z_k ~ N(0,1)
  ・時刻 t_k = k/n の過程値は  B_{t_k} = Σ_{j=1..k} ΔB_j  で近似できる。
本コードは上記をそのまま実装し，累積和で離散軌跡を描画するものである。
"""

import numpy as np
import matplotlib.pyplot as plt

# （グラフの日本語表記；ユーザー既定設定）------------------------------
from matplotlib import rcParams

rcParams["font.family"] = "IPAexGothic"
rcParams["font.sans-serif"] = "Meiryo"
# --------------------------------------------------------------------

# 乱数シード（再現性確保のため固定；必要に応じて変更する）
np.random.seed(4)

# ステップ数 n（区間 [0,1] を n 等分する）である
n = 1000

# Δt = 1/n の平方根。増分 ΔB_k ~ N(0, Δt) を  ΔB_k = sqrt(Δt) * Z_k で生成するための係数である
sqn = 1 / np.sqrt(n)

# Z_k ~ N(0,1) を n 個生成する（独立同分布）
z_values = np.random.randn(n)

# 初期値 B_0 = 0 として累積するための作業変数である
Yk = 0.0

# 離散軌跡を格納するリストである（本実装では長さ n；先頭に 0 を入れるならコメントを外す）
sb_motion = []
# sb_motion = [0.0]  # ← 初期値 0 を可視化に含めたい場合はこちらを用いる

# ブラウン運動の離散近似：B_{t_k} = Σ_{j=1..k} (sqn * Z_j)
for k in range(n):
    # 1 ステップ分の増分 ΔB_k = (1/√n) * Z_k を加える
    Yk = Yk + sqn * z_values[k]
    sb_motion.append(Yk)

# 参考：完全ベクトル化版（数値は上と一致するはずである）
# sb_motion_vec = np.cumsum(sqn * z_values)

# 描画設定：横軸をステップ番号として軌跡を可視化する
plt.figure(figsize=(8, 4))
plt.plot(sb_motion, linewidth=1.2)
plt.title("標準ブラウン運動の離散近似（n = {}）".format(n))
plt.xlabel("ステップ k")
plt.ylabel("過程値 B(t_k)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
