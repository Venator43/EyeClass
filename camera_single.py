# camera_single.py

import cv2
import math
import numpy as np
from sixdrepnet import SixDRepNet

class Camera():
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.model = SixDRepNet()
        # Opencvのカメラをセットします。(0)はノートパソコンならば組み込まれているカメラ

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()

        label = False
        pitch, yaw, roll = self.model.predict(image)
        if abs(roll) > 5:
            label = True
        self.model.draw_axis(image, yaw, pitch, roll)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes(), label, roll

        # read()は、二つの値を返すので、success, imageの2つ変数で受けています。
        # OpencVはデフォルトでは raw imagesなので JPEGに変換
        # ファイルに保存する場合はimwriteを使用、メモリ上に格納したい時はimencodeを使用
        # cv2.imencode() は numpy.ndarray() を返すので .tobytes() で bytes 型に変換


