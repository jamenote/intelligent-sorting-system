import cv2


def showvideo():
    cap=cv2.VideoCapture(0)
    while True:
        sucess, img=cap.read()
        cv2.imshow("img",img)
        k=cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    showvideo()