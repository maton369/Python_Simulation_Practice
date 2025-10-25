# NumPyライブラリをインポート
# 数値計算や剰余演算(np.mod)に利用する
import numpy as np

# 線形合同法 (Linear Congruential Generator: LCG) のパラメータ設定
# 一般式: X_{n+1} = (a * X_n + c) mod m
a = 75  # 乗数 (multiplier)
c = 0  # 加算定数 (increment) — 0の場合は「乗算型LCG」と呼ばれる
m = 2**31 - 1  # 法 (modulus) — Mersenne数 (2^31 - 1 = 2147483647)
x = 1  # 初期値 (seed)。

# 1から99まで繰り返し (合計99個の乱数を生成)
for i in range(1, 100):
    # LCGの基本式によって次の乱数を生成
    # 剰余演算を np.mod() で実行（xがmを超えた場合に折り返す）
    x = np.mod((a * x + c), m)

    # 生成された値を [0, 1) の範囲に正規化する
    u = x / m

    # 擬似乱数を出力
    print(u)
