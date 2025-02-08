import socket
import cv2
import numpy as np

# 创建 TCP/IP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定到指定的 IP 地址和端口
server_socket.bind(('0.0.0.0', 12345))  # '0.0.0.0' 表示绑定到所有可用的网络接口
server_socket.listen(1)

print("服务器启动，等待连接...")

# 等待客户端连接
client_socket, client_address = server_socket.accept()
print(f"客户端连接: {client_address}")

# 创建一个 OpenCV 窗口用于显示接收到的视频
cv2.namedWindow("Received Video")

while True:
    try:
        # 接收数据
        data = b""
        while len(data) < 921600:  # 假设每个图像大小为 640x480x3（RGB），你可以根据需要调整
            packet = client_socket.recv(4096)  # 每次读取 4KB
            if not packet:
                break
            data += packet

        # 如果没有收到数据，退出循环
        if not data:
            break

        # 将接收到的数据解码为图像
        nparr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 显示接收到的图像
        if frame is not None:
            cv2.imshow("Received Video", frame)

        # 如果按下 'q' 键，则退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"接收数据时发生错误: {e}")
        break

# 清理资源
client_socket.close()
server_socket.close()
cv2.destroyAllWindows()
