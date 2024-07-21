import socket
import json
import time
import coordinate
from april import getphoto_left
from april import getphoto_right

x_cache = 0
y_cache = 0


def sd(z_real):
    global x_cache
    global y_cache
    x, y = coordinate.getpoint()
    x_real = 1015+y
    y_real = -794+x
    x_cache = x_real
    y_cache = y_real
    x_real = str(x_real)
    y_real = str(y_real)
    z_real = str(z_real)
    print(x_real)
    print(y_real)
    bool = 0    # bool值用来定义处于发送状态还是接收状态，这里是先发送
    # 0x3F表示6轴运动，而0xFF表示8轴运动
    data = {"dsID":"www.hc-system.com.HCRemoteCommand","reqType":"AddRCC","emptyList":"1","instructions":
        [{"oneshot":"1","action":"10","m0":x_real,"m1":y_real,"m2":z_real,"m3":"-178","m4":"1","m5":"-174","ckStatus":"0X3F","speed":"80","delay":"0","smooth":"0"},]}
    # data2 = {"dsID": "www.hc-system.com.HCRemoteCommand", "reqType": "AddRCC", "emptyList": "0", "instructions": [{"oneshot": "1", "action": "4", "m0": "0", "m1": "0", "m2": "0", "m3": "0", "m4": "0", "m5": "0","ckStatus": "0X3F", "speed": "20", "delay": "0", "smooth": "0"}]}
    # data4 =  {"dsID": "HCRemoteMonitor","cmdType": "command","cmdData": ["rewriteDataList", "800", "6", "0", "50000", "10000", "0", "0", "0", "12000"]}
    if bool == 0:    # 客户端
        ip_addr = ("192.168.4.4", 502)    # 客户端绑定另一个电脑的ipv4地址，端口可换1024到50000以内的值
        client = socket.socket()
        client.connect(ip_addr)
        client.send(json.dumps(data).encode('utf-8'))  # 发送传输需要的数据
        print('已发送')


def down(x, y, z):
    x = str(x)
    y = str(y)
    z = str(z)
    bool = 0    # bool值用来定义处于发送状态还是接收状态，这里是先发送
    # 0x3F表示6轴运动，而0xFF表示8轴运动
    data = {"dsID":"www.hc-system.com.HCRemoteCommand","reqType":"AddRCC","emptyList":"1","instructions":
        [{"oneshot":"1","action":"10","m0":x,"m1":y,"m2":z,"m3":"-178","m4":"1","m5":"-174","ckStatus":"0X3F","speed":"80","delay":"0","smooth":"0"},]}
    # data2 = {"dsID": "www.hc-system.com.HCRemoteCommand", "reqType": "AddRCC", "emptyList": "0", "instructions": [{"oneshot": "1", "action": "4", "m0": "0", "m1": "0", "m2": "0", "m3": "0", "m4": "0", "m5": "0","ckStatus": "0X3F", "speed": "20", "delay": "0", "smooth": "0"}]}
    # data4 =  {"dsID": "HCRemoteMonitor","cmdType": "command","cmdData": ["rewriteDataList", "800", "6", "0", "50000", "10000", "0", "0", "0", "12000"]}
    if bool == 0:    # 客户端
        ip_addr = ("192.168.4.4", 502)    # 客户端绑定另一个电脑的ipv4地址，端口可换1024到50000以内的值
        client = socket.socket()
        client.connect(ip_addr)
        client.send(json.dumps(data).encode('utf-8'))  # 发送传输需要的数据
        print('已发送')


def open(status):
    status = str(status)
    bool = 0    # bool值用来定义处于发送状态还是接收状态，这里是先发送
    # 0x3F表示6轴运动，而0xFF表示8轴运动
    data = {"dsID":"www.hc-system.com.HCRemoteCommand","reqType":"AddRCC","emptyList":"1","instructions":[{"oneshot":"1","action":"200","type":"5","io_status":status,"point":"0","delay":"0"}]}
    data2 = {"dsID":"www.hc-system.com.RemoteMonitor","reqType":"command","cmdData":["clearAlarmRunNext"]}
    if bool == 0:    # 客户端
        ip_addr = ("192.168.4.4", 502)    # 客户端绑定另一个电脑的ipv4地址，端口可换1024到50000以内的值
        client = socket.socket()
        client.connect(ip_addr)
        client.send(json.dumps(data).encode('utf-8'))
        # client.send(json.dumps(data2).encode('utf-8'))# 发送传输需要的数据
        print('已发送')


def testst(point, status):
    point = str(point)
    status = str(status)
    bool = 0  # bool值用来定义处于发送状态还是接收状态，这里是先发送
    # 0x3F表示6轴运动，而0xFF表示8轴运动
    data = {"dsID":"www.hc-system.com.HCRemoteCommand","reqType":"AddRCC","emptyList":"1","instructions":
        [{"oneshot":"1","action":"200","type":"0","io_status":status,"point":point,"delay":"0"}]}
    if bool == 0:  # 客户端
        ip_addr = ("192.168.4.4", 502)  # 客户端绑定另一个电脑的ipv4地址，端口可换1024到50000以内的值
        client = socket.socket()
        client.connect(ip_addr)
        client.send(json.dumps(data).encode('utf-8'))
        # client.send(json.dumps(data2).encode('utf-8'))# 发送传输需要的数据
        print('已发送')


for i in range(0, 10):
    bool1 = getphoto_left(1)
    bool2 = getphoto_right(2)
    down(927.5, -28, 1110)
    time.sleep(1)
    testst(5, 1)
    time.sleep(0.8)
    down(927.5, -28, 940)
    time.sleep(2)
    testst(5, 0)
    time.sleep(0.8)
    down(927.5, -28, 1110)
    time.sleep(3)
    sd(1150)
    time.sleep(8)
    print(x_cache, y_cache)
    down(x_cache, y_cache, 590)
    time.sleep(5)
    testst(5, 1)
    time.sleep(0.8)
    down(x_cache, y_cache, 1150)
    time.sleep(5)
    down(927.5, -28, 1110)
    time.sleep(3)