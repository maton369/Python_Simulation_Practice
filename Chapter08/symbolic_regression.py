# ================================================
# gplearn による記号回帰（Symbolic Regression）デモである。
# 目的：f(x, y) = x^2 + y^2 を「式の形」で学習させ、構文木として同定する。
# 構成：
#   (1) 可視化用メッシュ生成と 3D サーフェス描画
#   (2) 学習/評価データの乱数生成（-1〜1 の一様分布）
#   (3) 記号回帰モデル SymbolicRegressor の学習と評価
# 備考：
#   - 記号回帰は「予測精度」と「式の簡潔さ（解釈性）」のトレードオフである。
#   - 本例はノイズ無し・低次の可分関数であるため、適切な探索条件で高い R^2 を達成しやすい。
# ================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import (
    Axes3D,
)  # 3D 軸（古典 API）。互換性のため残しているである
from gplearn.genetic import SymbolicRegressor  # 記号回帰の主モデルである

# （任意）再現性が必要なら乱数シードを固定するである
# np.random.seed(42)

# ------------------------------------------------
# (1) 可視化：f(x, y) = x^2 + y^2 の地形を 3D 表示するである
# ------------------------------------------------
# x, y の一様な格子を作る。arange は端点を含まないが、密度が十分なら問題ないである
x = np.arange(-1, 1, 1 / 10.0)
y = np.arange(-1, 1, 1 / 10.0)
x, y = np.meshgrid(x, y)  # 2D グリッドに展開するである

# 目的関数値をグリッド上で計算するである（ベクトル化により高速である）
f_values = x**2 + y**2

# 3D サーフェスで地形を可視化するである
fig = plt.figure()
ax = Axes3D(fig)  # 代替：fig.add_subplot(111, projection='3d') でも良いである
ax.plot_surface(x, y, f_values)  # カラーマップ等は省略（最小例のため）
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# ------------------------------------------------
# (2) データ生成：学習/評価データを独立に作るである
# ------------------------------------------------
# 学習用入力（100 サンプル, 2 特徴）。各特徴は [-1, 1] の一様である
input_train = np.random.uniform(-1, 1, 200).reshape(100, 2)
# 学習用ラベルは真の関数 f(x,y)=x^2+y^2 により合成するである
output_train = input_train[:, 0] ** 2 + input_train[:, 1] ** 2

# 評価用（テスト）も同様に独立サンプルを生成するである
input_test = np.random.uniform(-1, 1, 200).reshape(100, 2)
output_test = input_test[:, 0] ** 2 + input_test[:, 1] ** 2

# ------------------------------------------------
# (3) 記号回帰モデルの構築と学習である
# ------------------------------------------------
# 使用する関数集合（演算子）を指定するである
# 今回の真の式 x^2+y^2 は add と mul（自己積）で表現可能であるため、最小集合で十分である
function_set = ["add", "sub", "mul"]

# SymbolicRegressor のハイパーパラメータ説明である：
#   - population_size: 遺伝的プログラミングの母集団サイズ
#   - function_set   : 利用可能な演算子集合（構文木の内部ノード）
#   - generations    : 進化世代数（多いほど探索は進むが計算が増える）
#   - stopping_criteria: 早期停止のしきい値（ここでは R^2 相当の損失基準）
#   - p_crossover / p_subtree_mutation / p_hoist_mutation / p_point_mutation:
#       交叉・突然変異の確率（探索の多様性を制御）
#   - max_samples    : 学習で用いるサンプル率（0.9 なら 90% サブサンプリング；汎化に有利な場合がある）
#   - verbose        : 学習の進行表示
#   - parsimony_coefficient: 式の複雑さに対するペナルティ（過剰に長い式を抑制）
#   - random_state   : モデル内部 RNG のシード（再現性）
sr_model = SymbolicRegressor(
    population_size=1000,
    function_set=function_set,
    generations=10,
    stopping_criteria=0.001,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    parsimony_coefficient=0.01,
    random_state=1,
)

# モデルを学習データに当てはめるである（内部で構文木の進化探索が行われる）
sr_model.fit(input_train, output_train)

# 学習で得られた最良プログラム（構文木）を文字列表現で出力するである
# 注意：_program は内部属性であり将来バージョンで変更される可能性がある
print(sr_model._program)

# テストデータに対する決定係数 R^2 を表示するである（1 に近いほど良い）
print("R2:", sr_model.score(input_test, output_test))
