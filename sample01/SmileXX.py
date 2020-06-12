# -*- coding:utf-8 -*-
import sys
import cv2
import numpy as np
from PIL import Image

# 笑い〇〇
class WaraiXX(object):
  _ESC_KEY = 27     # Escキー
  _INTERVAL= 33     # 待ち時間

  _cc = None
  _face_img = None
  
  # コンストラクタ
  def __init__(self):
    self._cc = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    self._face_img = cv2.imread("img/1.png", -1)
    pass
  
  # 起動
  def run(self):
    # 画面キャプチャー開始
    cap = cv2.VideoCapture(0)
    if cap.isOpened() is False:
      print("can not open camera")
      return
    
    cur_x = 0
    cur_y = 0
    cur_w = 0
    cur_h = 0
    
    # ESCが押されるまでループ
    while True:
      # 画面キャプチャー
      ret, img = cap.read()
      height, width = img.shape[:2]
      if cur_w == 0:
        cur_w = width
        cur_h = height
      
      # キャプチャー画像から顔の位置を取得
      x, y, w, h = self.classification(img)
      if x > 0:
        cur_x = x
        cur_y = y
        cur_w = w
        cur_h = h
      
      # 顔の位置にフィルタ画像を配置
      faceImg = cv2.resize(self._face_img, (cur_w + 40, cur_h + 40))
      img = self.mergeImg(img, faceImg, (cur_x, cur_y - 40))
      img = cv2.resize(img, (int(width * 1.5), int(height * 1.5)))
      cv2.imshow('smile XX', img)

      # キー入力判定
      key = cv2.waitKey(self._INTERVAL)
      if key == self._ESC_KEY:
        break
      else:
        self.changeImg(key)

    # 後処理
    cap.release()
    cv2.destroyAllWindows()

  # キャプチャー画像から顔の位置を取得
  def classification(self, img):
    height, width = img.shape[:2]
    
    # 処理速度を上げるため画像を縮小、グレースケール化
    img = cv2.resize(img, (int(width*0.5), int(height*0.5)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 顔検出
    face_list = self._cc.detectMultiScale(img, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))
    for (x, y, w, h) in face_list:
      return x * 2, y * 2, w * 2, h * 2
    return 0, 0, 0, 0

  # 背景とフィルタ画像の合成
  def mergeImg(self, backImg, frontImg, location):
    frontImg_height, frontImg_width = frontImg.shape[:2]

    # 背景をPIL形式に変換
    backImg = cv2.cvtColor(backImg, cv2.COLOR_BGR2RGB)
    pil_backImg = Image.fromarray(backImg)
    pil_backImg = pil_backImg.convert('RGBA')

    # オーバーレイをPIL形式に変換
    frontImg = cv2.cvtColor(frontImg, cv2.COLOR_BGRA2RGBA)
    pil_frontImg = Image.fromarray(frontImg)
    pil_frontImg = pil_frontImg.convert('RGBA')

    # 画像を合成
    pil_tmp = Image.new('RGBA', pil_backImg.size, (255, 255, 255, 0))
    pil_tmp.paste(pil_frontImg, location, pil_frontImg)
    result_image = Image.alpha_composite(pil_backImg, pil_tmp)

    # OpenCV形式に変換
    return cv2.cvtColor(np.asarray(result_image), cv2.COLOR_RGBA2BGRA)
  
  # 画像切り替え
  def changeImg(self, key):
    try:
      key = chr(key)
      if key >= "1" or key <="5":
        # 押されたキーに該当する画像をフィルタに設定(1～5まで対応)
        self._face_img = cv2.imread("img/" + key + ".png", -1)
    except:
      pass

# メイン処理
if __name__ == '__main__':
  wariXX = WaraiXX()
  wariXX.run()
  