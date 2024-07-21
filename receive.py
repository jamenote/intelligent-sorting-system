import socket
import os
import json
import time


print('listening...')
ip_addr = ("192.168.4.4", 502)  # 绑定本地ipv4地址
server = socket.socket()
server.connect(ip_addr)
# server.bind(ip_addr)
# server.listen(5)  # 连接队列长度，不用管

while True:  # 尝试接收，如果失败则重试
    try:
        conn, addr = server.accept()
        file_msg = conn.recv(2048000000)  # 这个长度可以改为1024，和下面同时改
        print(file_msg)
        msg_data = json.loads(file_msg.decode('utf-8'))  # 注意编解码方式相同
        print(msg_data)

    except:
        pass