import os
import sys
import cv2

# 获取文件的路径，兼容打包后的 .exe 文件
if getattr(sys, 'frozen', False):  # 如果程序是打包后的 .exe
    base_path = sys._MEIPASS  # PyInstaller 打包后的临时路径
else:
    base_path = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录

# 设置 Haar Cascade 的路径
face_cascade_path = os.path.join(base_path, 'haarcascade_frontalface_default.xml')
print("Cascade file path:", face_cascade_path)

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    print("Error loading Haar Cascade file!")
    exit()
