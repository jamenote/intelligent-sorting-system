import PyQt5, sys
import cv2
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from UserInterface import Ui_MainWindow
from april import getphoto_left, getphoto_right
from getpar import get_parm


class MyGui(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("机械臂抓取界面")

    def show_information(self, ifo=""):
        self.textBrowser.setText(ifo)

    def daoru(self):
        reply = QMessageBox.information(self, "提示框", "正在拍照，请勿关闭界面", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        bool1 = getphoto_left(0)
        bool2 = getphoto_right(0)
        bool = bool1 & bool2
        if bool:
            reply = QMessageBox.information(self, "提示框", "已完成拍照", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        else:
            reply = QMessageBox.information(self, "提示框", "未能成功拍照请检查文件", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    def biaoding(self):
        try:
            get_parm()
            reply = QMessageBox.information(self, "提示框", "已创建相机内参文件", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        except:
            reply = QMessageBox.information(self, "提示框", "未能创建内参请检擦文件", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    def bofang(self):
        cap = cv2.VideoCapture(0)
        while True:
            flag, image = cap.read()  # 从视频流中读取
            show = cv2.resize(image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                     QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    app = PyQt5.QtWidgets.QApplication(sys.argv)
    MyUiStart = MyGui()
    MyUiStart.show()
    sys.exit(app.exec_())
