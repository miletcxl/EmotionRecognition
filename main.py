import cv2
import numpy as np
from keras.models import load_model
import sys
from tensorflow.keras.preprocessing import image

# 加载预训练的情绪识别模型
model = load_model('emotion_recognition_model.keras')

# 设置情绪标签
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 加载OpenCV的人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Error loading Haar Cascade file!")
    sys.exit()

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 0表示默认摄像头

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 对每个检测到的人脸进行处理
    for (x, y, w, h) in faces:
        # 获取人脸图像并进行预处理
        face = frame[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
        face = cv2.resize(face, (48, 48))  # 调整尺寸至 48x48
        face = face / 255.0  # 归一化
        face = np.expand_dims(face, axis=-1)  # 增加一个维度 (48, 48, 1)
        face = np.expand_dims(face, axis=0)  # 增加 batch 维度 (1, 48, 48, 1)

        # 使用模型预测情绪
        prediction = model.predict(face)
        max_index = np.argmax(prediction[0])
        predicted_emotion = emotion_labels[max_index]

        # 绘制人脸和情绪标签
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Emotion Recognition', frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
