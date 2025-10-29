# -*- coding: utf-8 -*-
"""
一次元セル・オートマトン（Elementary Cellular Automaton; ECA）の実装例である。
Wolfram の規則番号（0〜255）を 8 ビットの2進表現に変換し、3セル近傍
（左・中央・右）の状態から次時刻の状態を決める。ここではルール126を既定とする。

【要点】
- 近傍の3ビットを 4*L + 2*C + 1*R で 0..7 の整数に符号化する。
- Wolfram のビット列は MSB が近傍 111（=7）、LSB が 000（=0）に対応する。
  よってインデックスは bin_rule[7 - code] として MSB→LSB の向きを合わせる。
- 周期境界（トーラス）を np.roll で実現する。
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1) グリッドとルールの設定である
# -------------------------------
cols_num = 100  # 横方向セル数である
rows_num = 100  # 時刻（縦方向行数＝世代数）である

wolfram_rule = 126  # 例：ルール126である（任意に変更可能）
# ルール番号を 8 ビットの配列（MSB→LSB）に展開する
bin_rule = np.array(
    [int(b) for b in np.binary_repr(wolfram_rule, width=8)], dtype=np.int8
)
print("Binary rule (MSB→LSB) :", bin_rule)

# -------------------------------
# 2) 初期状態の生成である
# -------------------------------
# 全体の状態配列（行=時間、列=セル位置）を 0 で初期化する
cell_state = np.zeros((rows_num, cols_num), dtype=np.int8)

# 例：初期行をランダムに与える（0/1 を一様に）
#    再現性が必要なら np.random.seed(seed) を用いるとよい
cell_state[0, :] = np.random.randint(0, 2, cols_num, dtype=np.int8)

# 例：中央1セルのみ1にしたい場合（グライダ様パターン観察に有用）
# cell_state[0, cols_num // 2] = 1

# -------------------------------
# 3) 近傍の符号化と時間発展である
# -------------------------------
# 3セル近傍を 4*L + 2*C + 1*R で符号化するための重み列である
update_window = np.array([[4], [2], [1]], dtype=np.int8)

for t in range(rows_num - 1):
    # 周期境界条件（端は反対側に巻く）で近傍（3×cols）の配列を作る
    left = np.roll(cell_state[t, :], +1)  # 左隣
    cent = cell_state[t, :]  # 中央
    right = np.roll(cell_state[t, :], -1)  # 右隣
    update = np.vstack((left, cent, right)).astype(np.int8)

    # 近傍3ビットの符号化（0..7）である
    code = np.sum(update * update_window, axis=0).astype(np.int8)

    # Wolfram 規則に基づく更新：
    # bin_rule の 0番目は 111 に対応するため、インデックスを (7 - code) とする
    cell_state[t + 1, :] = bin_rule[7 - code]

# -------------------------------
# 4) 可視化である
# -------------------------------
plt.figure(figsize=(8, 6))
# 0/1 を白黒で描画する。補間を切って正方画素を保つ
plt.imshow(cell_state, cmap=plt.cm.binary, interpolation="nearest", aspect="auto")
plt.title(f"Elementary Cellular Automaton (Rule {wolfram_rule})")
plt.xlabel("cell index (x)")
plt.ylabel("time step (t)")
plt.tight_layout()
plt.show()
