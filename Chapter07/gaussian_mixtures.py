# ------------------------------------------------------------
# ガウス混合分布（Gaussian Mixture Model; GMM）の1次元デモである
# ・平均が異なる2つの正規分布 N(mean_1, st_1^2), N(mean_2, st_2^2) から
#   疑似データを生成し、ヒストグラムと KDE を可視化するである
# ・scikit-learn の GaussianMixture により混合分布を当てはめ、
#   推定された平均・分散およびクラスタラベルを用いて結果を検証するである
# ・ラベル 0/1 は GMM の「任意の付番」であり、真の生成分布の 0/1 と必ずしも一致しない点に注意である
#   （= ラベルスイッチング問題）である
# ------------------------------------------------------------

import os, pathlib, sys
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd

# （任意）日本語フォントを IPA 系にしたい場合の設定例である
# 文字フォント設定：IPA 系を最優先にし、無ければ日本語フォントのフォールバックを用いるである
from matplotlib import rcParams, font_manager
from matplotlib.font_manager import FontProperties

# 1) IPA系フォント候補（Homebrew経由でよく置かれる場所も含む）である
ipa_files = [
    # 代表的なファイル名
    "ipaexg.ttf",
    "ipag.ttf",
    "ipagp.ttf",
    "ipaexm.ttf",
    "ipam.ttf",
    "ipamp.ttf",
]
search_dirs = [
    os.path.expanduser("~/Library/Fonts"),
    "/Library/Fonts",
    "/System/Library/Fonts",
    "/opt/homebrew/share/fonts",
    "/usr/local/share/fonts",
]

found_path = None
for d in search_dirs:
    if os.path.isdir(d):
        for name in ipa_files:
            p = os.path.join(d, name)
            if os.path.exists(p):
                found_path = p
                break
    if found_path:
        break

# 2) 見つかったIPAフォントを FontManager に明示登録して使うである
if found_path:
    font_manager.fontManager.addfont(found_path)
    # ファイル名から妥当なファミリ名を推定（ipaexg.ttf → IPAexGothic など）
    family_guess = (
        "IPAexGothic" if "g" in os.path.basename(found_path).lower() else "IPAexMincho"
    )
    chosen_family = family_guess
else:
    # 見つからないときは macOS 標準日本語フォントへフォールバックである
    chosen_family = "Hiragino Sans"

# 3) フォントキャッシュを再構築して新規フォントを認識させるである
#    （Matplotlib 3.6+ でも動く簡易リフレッシュ）
font_manager._load_fontmanager(try_read_cache=False)

# 4) IPAを最優先に設定。Arial等に落ちないよう sans-serif リストも上書きするである
rcParams["font.family"] = [chosen_family]
rcParams["font.sans-serif"] = [
    chosen_family,
    "Hiragino Sans",
    "Yu Gothic",
    "Noto Sans CJK JP",
    "Meiryo",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False

# 5) Seabornでフォントが上書きされないよう、明示的に指定するである
sns.set_theme(style="white", font=chosen_family)

# 6) 実際に解決したか確認（Matplotlibが選んだフォントの実体パスを表示）である
chosen_path = font_manager.findfont(chosen_family, fallback_to_default=False)
print("Using font.family:", chosen_family)
print("Font file path   :", chosen_path)
print("Cache dir        :", mpl.get_cachedir())
fp = FontProperties(fname=chosen_path)

# ------------------------------------------------------------
# 1) 2つの正規分布の母数（平均・標準偏差）を設定するである
#    mean_1, st_1：第1モード、mean_2, st_2：第2モード
# ------------------------------------------------------------
mean_1 = 25
st_1 = 9
mean_2 = 50
st_2 = 5

# ------------------------------------------------------------
# 2) サンプル生成である
#    ・第1分布から 3000 点、第2分布から 7000 点生成するである
#    ・ここでは乱数シードは固定していない（再現性が必要なら np.random.seed を用いる）である
# ------------------------------------------------------------
n_dist_1 = np.random.normal(loc=mean_1, scale=st_1, size=3000)
n_dist_2 = np.random.normal(loc=mean_2, scale=st_2, size=7000)

# 2つを結合して混合データを得るである（混合比は 0.3 : 0.7）
dist_merged = np.hstack((n_dist_1, n_dist_2))

# ------------------------------------------------------------
# 3) データの外観をヒストグラムと KDE で確認するである
# ------------------------------------------------------------
sns.set_style("white")
sns.histplot(data=dist_merged, kde=True)
plt.title("混合分布のヒストグラムとKDE（観測データ）", fontproperties=fp)
plt.xlabel("x", fontproperties=fp)
plt.ylabel("frequency", fontproperties=fp)
plt.show()

# ------------------------------------------------------------
# 4) GMM による当てはめである
#    ・特徴量は (N,1) 形状に整形が必要である
#    ・init_params='kmeans' で初期クラスタ中心を k-means によって初期化するである
# ------------------------------------------------------------
dist_merged_res = dist_merged.reshape((len(dist_merged), 1))
gm_model = GaussianMixture(n_components=2, init_params="kmeans")
gm_model.fit(dist_merged_res)

# 推定結果の表示である
print(f"Initial distribution means = {mean_1, mean_2}")
print(f"Initial distribution standard deviation = {st_1, st_2}")

# gm_model.means_ は形状 (n_components, n_features) であり、ここでは (2,1) である
print(f"GM_model distribution means = {gm_model.means_}")

# gm_model.covariances_ は既定 'full' のため (n_components, n_features, n_features) である
# 1次元ゆえ各成分は 1×1 行列であり、要素の平方根をとると標準偏差の推定値になるである
print(f"GM_model distribution standard deviation = {np.sqrt(gm_model.covariances_)}")

# ------------------------------------------------------------
# 5) 学習済みGMMにより各サンプルのクラスタラベルを推定するである
#    ・ここでのラベル 0/1 は GMM 内部で任意に割り振られる点に注意である
# ------------------------------------------------------------
dist_labels = gm_model.predict(dist_merged_res)

# 推定ラベルによる色分けヒストグラムである（モデルが分離した2クラスタの可視化）
sns.set_style("white")
data_pred = pd.DataFrame({"data": dist_merged, "label": dist_labels})
sns.histplot(data=data_pred, x="data", kde=True, hue="label")
plt.title("GMM 推定ラベルによるヒストグラム（モデル推定）", fontproperties=fp)
plt.xlabel("x", fontproperties=fp)
plt.ylabel("frequency", fontproperties=fp)
leg = plt.gca().get_legend()
if leg is not None:
    for t in leg.get_texts():
        t.set_fontproperties(fp)
    if leg.get_title() is not None:
        leg.get_title().set_fontproperties(fp)
plt.show()

# ------------------------------------------------------------
# 6) 真のラベル（生成時点の所属）での可視化である
#    ・先頭3000点を 0（第1分布）、残り7000点を 1（第2分布）として付与したものである
#    ・GMMのラベルと一致する保証は無い（= ラベルスイッチング）である
# ------------------------------------------------------------
label_0 = np.zeros(3000, dtype=int)
label_1 = np.ones(7000, dtype=int)
labels_merged = np.hstack((label_0, label_1))
data_init = pd.DataFrame({"data": dist_merged, "label": labels_merged})

sns.set_style("white")
sns.histplot(data=data_init, x="data", kde=True, hue="label")
plt.title("生成時の真のラベルによるヒストグラム（グラウンドトゥルース）", fontproperties=fp)
plt.xlabel("x", fontproperties=fp)
plt.ylabel("frequency", fontproperties=fp)
leg = plt.gca().get_legend()
if leg is not None:
    for t in leg.get_texts():
        t.set_fontproperties(fp)
    if leg.get_title() is not None:
        leg.get_title().set_fontproperties(fp)
plt.show()

# ------------------------------------------------------------
# 7) 補足である
#    ・推定ラベルと真のラベルの対応を取りたい場合は、平均値順に並べ替えてマッピングするか、
#      混同行列（sklearn.metrics.confusion_matrix）を用いて最適対応を決めるとよいである
#    ・分散パラメータの形状は covariance_type に依存する（'full','diag','tied','spherical'）
#      1次元で分散のスカラーを直接得たい場合は covariance_type='diag' を選ぶのも一案である
# ------------------------------------------------------------
