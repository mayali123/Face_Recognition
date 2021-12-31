# -*- coding: UTF-8 -*-
# @Time :  16:41
# @Author :mayali123
# @File : Face_recognition_GUI.py
# @Software : PyCharm


from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QPushButton, QMessageBox, QDialog
from PyQt5.QtCore import pyqtSignal
import sys
import time
import os
from Face_recognition import *

class input_face(QDialog):
    #  信号函数
    # 关闭 的 信号
    closemeg = pyqtSignal(object)

    def __init__(self, parent=None):
        super(input_face, self).__init__(parent)
        self.Face_recognition = Face_recognition()
        self.set_ui()
        # 定时器
        self.timer_camera = QtCore.QTimer()  # 初始化定时器
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # 初始化摄像头
        self.connect()
        self.timer_camera.start(100)

        self.cnt = 1

    # ui 界面
    def set_ui(self):
        # 窗口
        self.setFixedSize(720, 510)
        self.setWindowIcon(QIcon("./Resources/2.jpg"))
        self.setWindowTitle(u'录入人脸')

        # 摄像头
        self.camera = QtWidgets.QLabel(parent=self)
        self.camera.move(40, 60)
        self.camera.setFixedSize(340, 270)

        # 提取人脸
        self.face = QtWidgets.QLabel(parent=self)
        self.face.move(430, 60)
        self.face.setFixedSize(184, 224)

        # 显示文字
        self.label = QtWidgets.QLabel(parent=self, text='请输入学号/英文名')
        self.label.move(100, 354)
        self.label.setFixedSize(130, 15)

        # 摄像头
        self.label1 = QtWidgets.QLabel(parent=self, text='摄像头')
        self.label1.move(40, 43)

        # 人脸
        self.label2 = QtWidgets.QLabel(parent=self, text='人脸')
        self.label2.move(430, 43)

        # 输入框
        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.move(238, 352)
        self.lineEdit.setFixedSize( 113, 21)

        # 显示按钮
        self.Photograph = QPushButton(parent=self, text='拍照')
        self.Photograph.move(410, 350)
        self.Photograph.setFixedSize(92, 28)

        # 显示信息的文本框
        self.textBrowser = QtWidgets.QTextBrowser(self)
        self.textBrowser.move(20, 390)
        self.textBrowser.setFixedSize(680, 100)

    # 建立通信连接
    def connect(self):
        self.timer_camera.timeout.connect(self.show_camera)
        self.Photograph.clicked.connect(self.show_face)

    # 摄像头开始显示
    def show_camera(self):
        flag, self.frame = self.cap.read()

        self.image, self.x, self.y, self.w, self.h =  self.Face_recognition.face_detection(self.frame.copy())

        show = cv.resize(self.image, (340, 270))
        show = cv.cvtColor(show, cv.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    # 关闭摄像头
    def close_camera(self):
        # 关闭摄像头
        if self.cap.isOpened():
            self.cap.release()
            print('关闭摄像头！！！')
        # 关闭定时器
        if self.timer_camera.isActive():
            self.timer_camera.stop()

    # 截取人脸
    def show_face(self):
        self.input_id = self.lineEdit.text()

        if len(self.input_id) != 0 and self.w != 0 and self.h != 0:
            self.lineEdit.setDisabled(True)
            # 选择一块人脸的区域
            selective_img = self.frame[self.y:self.y + self.h + 15, self.x:self.x + self.w]

            show = cv.resize(selective_img, (184, 224))
            show = cv.cvtColor(show, cv.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.face.setPixmap(QtGui.QPixmap.fromImage(showImage))

            # 变成黑白图
            selective_img = cv.cvtColor(selective_img, cv.COLOR_BGR2GRAY)
            # 缩小尺寸
            selective_img = cv.resize(selective_img, (92, 112))

            save_path = './orl/' + str(self.input_id)
            # 如果文件夹 存在则不创建新的 文件夹
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            save_path = './orl/' + self.input_id + '/' + str(self.cnt) + '.pgm'
            # 写入
            cv.imwrite(save_path, selective_img)

            # 输出
            self.put_text("文件已经写入{}\n还需要拍{}张\n".format(save_path, 10-self.cnt))

            self.cnt += 1
        elif len(self.input_id)==0:
            self.put_text("拍照失败！请输入学号/英文名")
        else:
            self.put_text("拍照失败！未识别出人脸，请调整姿势重新拍")
        if self.cnt > 10:
            choice = QMessageBox.warning(None, "关闭", "已经采集10张照片是否需要继续采集", QMessageBox.No | QMessageBox.Yes)
            if choice == QMessageBox.No:
                # 关闭摄像头
                self.close_camera()
                # 关闭 界面
                self.close()
            else:
                self.lineEdit.setDisabled(False)
                self.cnt = 1

    # 显示文字
    def put_text(self, text):
        self.textBrowser.append(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ':\n' \
            + text)

    # # 重新 关闭 事件
    def closeEvent(self, QCloseEvent):
        self.closemeg.emit(None)
        print("关闭")
        QCloseEvent.accept()


if __name__ == '__main__':
    App = QtWidgets.QApplication(sys.argv)
    win = input_face()
    win.show()
    sys.exit(App.exec_())