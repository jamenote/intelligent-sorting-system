import time
import cv2


def getphoto_left(id):
    cap = cv2.VideoCapture(id, cv2.CAP_DSHOW)
    flag = cap.isOpened()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while (flag):
        time.sleep(2)
        ret, frame = cap.read()
        # cv2.imshow("Video", frame)
        k = cv2.waitKey(1) & 0xFF
        time.sleep(2)
        frame = cv2.resize(frame, (1920, 1080))
        bool = cv2.imwrite("C:/Users/11478/Desktop/par/0-left.jpg", frame)
        print("save C:/Users/11478/Desktop/par/0-left.jpg successfuly!")
        print("-------------------------")
        break
    cap.release() # 释放摄像头
    cv2.destroyAllWindows()# 释放并销毁窗口
    return bool


def getphoto_right(id):
    cap = cv2.VideoCapture(id, cv2.CAP_DSHOW)
    flag = cap.isOpened()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while (flag):
        time.sleep(2)
        ret, frame = cap.read()
        k = cv2.waitKey(1) & 0xFF
        time.sleep(2)
        frame = cv2.resize(frame, (1920, 1080))
        bool = cv2.imwrite("C:/Users/11478/Desktop/par/0-right.jpg", frame)
        print("save C:/Users/11478/Desktop/par/0-right.jpg successfuly!")
        print("-------------------------")
        break
    cap.release() # 释放摄像头
    cv2.destroyAllWindows()# 释放并销毁窗口
    return bool


# bool1 = getphoto_left(0)
# bool2 = getphoto_right(0)
# print(bool1)
# print(bool2)



