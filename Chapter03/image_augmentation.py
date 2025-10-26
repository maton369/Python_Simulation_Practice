# ------------------------------------------------------------
# TensorFlow Keras ImageDataGenerator を用いた画像データ拡張（Data Augmentation）
# ------------------------------------------------------------
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------------------------------------------
# 1. データ拡張パラメータの設定
# ------------------------------------------------------------
# 画像をランダムに変換してデータを水増しする。
# モデルの汎化性能を高めるために使用される。
image_generation = ImageDataGenerator(
    rotation_range=10,  # 回転角度（±10度）
    width_shift_range=0.1,  # 横方向シフト（全体幅の10%）
    height_shift_range=0.1,  # 縦方向シフト（全体高さの10%）
    shear_range=0.1,  # シアー（せん断）変換
    zoom_range=0.1,  # ズーム倍率（±10%）
    horizontal_flip=True,  # 左右反転
    fill_mode="nearest",  # 変換後の空白を最近傍ピクセルで補完
)

# ------------------------------------------------------------
# 2. 元画像の読み込みと前処理
# ------------------------------------------------------------
# 対象画像をロード（例：colosseum.jpg）
source_img = load_img("colosseum.jpg")

# PIL形式 → NumPy配列に変換
x = img_to_array(source_img)

# ImageDataGenerator は (batch_size, height, width, channels) の形を期待するため、
# 先頭に1次元（バッチ次元）を追加
x = x.reshape((1,) + x.shape)

# ------------------------------------------------------------
# 3. データ拡張と保存
# ------------------------------------------------------------
# flow() メソッドを使ってバッチ単位で拡張画像を生成。
# 生成した画像は指定ディレクトリに自動保存される。
i = 0
for batch in image_generation.flow(
    x,
    batch_size=1,
    save_to_dir="AugImage",  # 保存先ディレクトリ
    save_prefix="new_image",  # ファイル名の接頭辞
    save_format="jpeg",  # 画像形式
):
    i += 1
    if i > 50:
        # 50枚生成したらループを終了
        break

# ------------------------------------------------------------
# 実行後:
# "AugImage/" フォルダに new_image_001.jpeg 〜 new_image_050.jpeg が生成される。
# 各画像には回転・平行移動・拡大縮小・反転などの変換がランダムに適用される。
# ------------------------------------------------------------
