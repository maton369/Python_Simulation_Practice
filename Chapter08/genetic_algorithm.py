# ------------------------------------------------------------
# 実数コーディングGA（遺伝的アルゴリズム）の最小実装である。
# 目的：線形目的関数 y = sum_i (x_i * var_values[i]) を最大化する。
# 制約：各遺伝子（係数） x_i は区間 [-10, 10] にある（初期個体はこの範囲で一様乱数）。
#
# 流れ：
#   1) 初期母集団の生成
#   2) 適応度（目的値）の計算
#   3) エリート選択（上位 sel_rate 個体を親集合 par_sel に複製）
#   4) 1点交叉（遺伝子配列の前半/後半を親2体から継承）で子個体を生成
#   5) 一様変異（任意の1遺伝子に [-1,1] のノイズを加算）
#   6) 次世代を構成（エリート保持＋子個体）
#   7) 2)〜6) を num_gen 世代繰り返し
#
# 備考：
#  - 本タスクは「線形目的＋箱拘束」のため解析解は x_i = 10*sign(var_values[i]) である。
#    GA の動作確認や教育目的の例として捉えるとよい。
#  - 変異後にクリップ（np.clip）を行うと制約違反を防げるが、ここでは元コードの挙動を保持する。
# ------------------------------------------------------------

import numpy as np

# 目的関数の重み（係数ベクトル）である。長さは num_coeff と一致させること。
var_values = [1, -3, 4.5, 2]

# 遺伝子長（= 変数の数）、個体数、選択する親の数（エリート数）である。
num_coeff = 4
pop_chrom = 10
sel_rate = 5

# 初期母集団の生成：形状 (個体数, 遺伝子長) の実数配列を [-10, 10] で一様サンプリングする。
pop_size = (pop_chrom, num_coeff)
pop_new = np.random.uniform(low=-10.0, high=10.0, size=pop_size)
print(pop_new)

# 進化させる総世代数である。
num_gen = 100
for k in range(num_gen):
    # 適応度（目的関数値）の計算：各個体ベクトルと var_values の要素積をとり、遺伝子次元で総和。
    # ここでは「最大化」を目標とする。
    fitness = np.sum(pop_new * var_values, axis=1)

    # 親集合（エリート）の入れ物を確保する。形状は (選択数, 遺伝子長)。
    par_sel = np.empty((sel_rate, pop_new.shape[1]))

    # 進捗ログ：現世代番号と、現状のベスト適応度を表示する。
    print("Current generation = ", k)
    print("Best fitness value : ", np.max(fitness))

    # エリート選択：適応度最大の個体を 1 つずつ取り出し、par_sel に複製する。
    # 同値が複数ある場合は先頭のインデックスを採用する。
    # 取り出した個体は fitness を最小値に潰して再選択を防ぐ（簡易な上位 sel_rate 抽出の実装）。
    for i in range(sel_rate):
        sel_id = np.where(
            fitness == np.max(fitness)
        )  # 現在の最大値のインデックス集合を取得
        sel_id = sel_id[0][0]  # 先頭の1つを採用
        par_sel[i, :] = pop_new[sel_id, :]  # 親集合にコピー
        fitness[sel_id] = np.min(fitness)  # 同じ個体が選ばれないよう最小値に置換

    # 子個体数と器の準備：全個体数からエリート数を引いた分だけ子を作る。
    offspring_size = (pop_chrom - sel_rate, num_coeff)
    offspring = np.empty(offspring_size)

    # 交叉の分割点（1点交叉の位置）を決める。ここでは配列の中央に固定している。
    crossover_lenght = int(offspring_size[1] / 2)

    # 交叉：各子個体について、親2体をランダムサンプリングし、
    #  先頭側は親1、後半側は親2から遺伝子を継承して生成する。
    for j in range(offspring_size[0]):
        par1_id = np.random.randint(0, par_sel.shape[0])
        par2_id = np.random.randint(0, par_sel.shape[0])
        offspring[j, 0:crossover_lenght] = par_sel[par1_id, 0:crossover_lenght]
        offspring[j, crossover_lenght:] = par_sel[par2_id, crossover_lenght:]

    # 変異：各子個体について、ランダムに選んだ1遺伝子に [-1, 1] の一様乱数ノイズを加える。
    # 注意：ここでは区間 [-10,10] のクリップをしていないため、境界を超える場合がある。
    for m in range(offspring.shape[0]):
        mut_val = np.random.uniform(-1.0, 1.0)  # 変異量
        mut_id = np.random.randint(0, par_sel.shape[1])  # 変異させる遺伝子の位置
        offspring[m, mut_id] = offspring[m, mut_id] + mut_val

    # 次世代の構成：先頭にエリート（親集合）をそのまま保持し、残りを子個体で埋める（エリート戦略）。
    pop_new[0 : par_sel.shape[0], :] = par_sel
    pop_new[par_sel.shape[0] :, :] = offspring

# 進化終了後：最終世代の適応度を計算し、最大値の個体（近似解）を報告する。
fitness = np.sum(pop_new * var_values, axis=1)
best_id = np.where(fitness == np.max(fitness))
print("Optimized coefficient values = ", pop_new[best_id, :])
print("Maximum value of y = ", fitness[best_id])

# ------------------------------------------------------------
# 参考（理論上の最適解）：
#  線形目的かつ箱拘束なので、最良は各成分で境界に貼り付く。
#   x*_i = 10  (var_values[i] > 0)
#   x*_i = -10 (var_values[i] < 0)
#  本例では [10, -10, 10, 10]、目的値 105.0 が解析解である。
#  GA の結果がこの値にどれだけ近づくかで動作を検証するとよい。
# ------------------------------------------------------------
