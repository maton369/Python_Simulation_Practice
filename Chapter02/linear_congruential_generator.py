# NumPyライブラリをインポート
# np.mod()（剰余演算）などの数学関数を使用するため
import numpy as np

# 線形合同法 (Linear Congruential Generator, LCG) のパラメータ設定
# 乱数生成の基礎となる式: X_{n+1} = (a * X_n + c) mod m
a = 2  # 乗数 (multiplier)
c = 4  # 加算定数 (increment)
m = 5  # 法 (modulus)
x = 3  # 初期値 (seed)

# 1 から 16 まで繰り返し（16個の乱数を生成）
for i in range(1, 17):
    # LCGの再帰式に基づいて次の乱数を生成
    x = np.mod((a * x + c), m)
    
    # 計算結果を出力（各ステップの擬似乱数を表示）
    print(x)