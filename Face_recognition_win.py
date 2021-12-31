# -*- coding: UTF-8 -*-
# @Time :  16:41
# @Author :mayali123
# @File : Face_recognition_GUI.py
# @Software : PyCharm


from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QPushButton, QMessageBox , QDialog
from PyQt5.QtCore import pyqtSignal
import sys

from Face_recognition import *

class Face_recognition_win(QDialog):
    #  信号函数
    # 关闭 的 信号
    closemeg = pyqtSignal(object)

    def __init__(self, model, parent=None):
        # print('1111')
        super(Face_recognition_win, self).__init__(parent)
        self.timer_camera = QtCore.QTimer()  # 初始化定时器
        self.cap = cv.VideoCapture()  # 初始化摄像头
        self.CAM_NUM = 0
        self.set_ui()

        self.connect()
        self.__flag_work = 0
        self.x = 0
        self.count = 0

        # 得到 人脸识别
        self.Face_recognition = Face_recognition(svm_model = model)

    # ui 界面
    def set_ui(self):
        # 窗口
        self.setFixedSize(660, 530)
        self.setWindowIcon(QIcon("./Resources/2.jpg"))
        self.setWindowTitle(u'人脸识别')

        # 显示视频
        self.label_show_camera = QtWidgets.QLabel(parent=self)
        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.move(10, 5)
        self.label_show_camera.setAutoFillBackground(False)

        # 按钮
        self.button_open_camera = QPushButton(parent=self, text='打开摄像头')
        self.button_open_camera.move(195, 492)
        # print(self.button_open_camera.size())

        self.button_close = QPushButton(parent=self, text='退出')
        self.button_close.move(405, 492)

    # 建立通信连接
    def connect(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.button_close.clicked.connect(self.close)

    # 打开摄像头按钮的控制
    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM, cv.CAP_DSHOW)
            if flag == False:
                msg = QtWidgets.QMessageBox.Warning(self, u'Warning', u'请检测相机与电脑是否连接正确',
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
                # if msg==QtGui.QMessageBox.Cancel:
                #                     pass
            else:
                self.timer_camera.start(30)
                self.button_open_camera.setText(u'关闭摄像头')
        else:
            # 关闭摄像头
            self.close_camera()
            self.button_open_camera.setText(u'打开摄像头')

    # 摄像头开始显示
    def show_camera(self):
        flag, self.image = self.cap.read()

        self.image = self.Face_recognition.face_detection_identify(self.image)

        show = cv.resize(self.image, (640, 480))
        show = cv.cvtColor(show, cv.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    # 关闭摄像头
    def close_camera(self):
        # 关闭摄像头
        if self.cap.isOpened():
            self.cap.release()
            self.label_show_camera.clear()
            print('关闭摄像头！！！')
        # 关闭定时器
        if self.timer_camera.isActive():
            self.timer_camera.stop()

    # 重新 关闭 事件
    def closeEvent(self, QCloseEvent):
        print("进入closeEvent")
        choice = QMessageBox.warning(None, "关闭", "是否关闭！", QMessageBox.Yes | QMessageBox.Cancel)
        if choice == QMessageBox.Yes:

            self.closemeg.emit(None)
            # 关闭摄像头
            self.close_camera()
            # 关闭 界面
            self.close()
        else:
            QCloseEvent.ignore()

    # 重写 键盘 按下事件
    def keyPressEvent(self, QKeyEvent):
        if QKeyEvent.key() == QtCore.Qt.Key_Q:
            self.close_camera()


if __name__ == '__main__':
    App = QtWidgets.QApplication(sys.argv)
    win = Face_recognition_win()
    win.show()
    sys.exit(App.exec_())