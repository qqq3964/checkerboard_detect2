import cv2
import tcar
import numpy as np

img = cv2.imread('data/000045.png', cv2.IMREAD_GRAYSCALE)
color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 점 색상 위해 컬러로 변환

corners = tcar.get_corners(img, cols=8, rows=6)  # corners: (N, 2) numpy

for pt in corners:
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(color_img, (x, y), radius=1, color=(0, 0, 255), thickness=-1)

# 4. 결과 보기
cv2.imshow('Detected Chessboard Corners', color_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
