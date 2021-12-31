from PyQt5.QtGui import  QFont
from input_face import *
from Train_svm_model import *
from Face_recognition_win import *

class Main_win(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Main_win, self).__init__(parent)
        self.set_ui()
        self.connect()


    # ui 界面
    def set_ui(self):
        # 窗口
        self.setFixedSize(480, 320)
        self.setWindowIcon(QIcon("./Resources/2.jpg"))
        self.setWindowTitle(u'人脸识别')

        # 显示一段话
        self.label = QtWidgets.QLabel(parent=self, text='welcome to face app!')
        self.label.move(60, 70)
        self.label.setFixedSize(361, 111)

        # 设置字体
        font = QFont()
        font.setFamily("Ink Free")
        font.setPointSize(20)
        self.label.setFont(font)
        # 居中
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        # 显示按钮
        self.input_face_pushbutton = QPushButton(parent=self, text='人脸采集')
        self.input_face_pushbutton.move(30, 230)
        self.input_face_pushbutton.setFixedSize(120, 45)

        self.train_model_pushbutton = QPushButton(parent=self, text='训练模型')
        self.train_model_pushbutton.move(180, 230)
        self.train_model_pushbutton.setFixedSize(120, 45)


        self.face_recognition_pushbutton = QPushButton(parent=self, text='人脸识别')
        self.face_recognition_pushbutton.move(330, 230)
        self.face_recognition_pushbutton.setFixedSize(120, 45)
        self.face_recognition_pushbutton.setDisabled(True)

    # 建立通信连接
    def connect(self):
        self.input_face_pushbutton.clicked.connect(self.input_face_click)
        self.train_model_pushbutton.clicked.connect(self.train_model_click)
        self.face_recognition_pushbutton.clicked.connect(self.face_recognition_click)

    # 点击录入人脸按钮
    def input_face_click(self):
        sec = input_face()
        self.setVisible(False)
        sec.closemeg.connect(lambda :(self.setVisible(True),
                             self.train_model_pushbutton.setDisabled(False)
                                      )
                             )
        sec.exec_()

    # 点击训练按钮
    def train_model_click(self):
        self.svm_model = Train_svm_model('./orl')
        QMessageBox.information(self, "训练完成", u'SVM模型已经训练完成')
        self.face_recognition_pushbutton.setDisabled(False)


    # 点击人脸识别按钮
    def face_recognition_click(self):
        sec = Face_recognition_win(model=self.svm_model)
        self.setVisible(False)
        sec.closemeg.connect(lambda: self.setVisible(True))
        sec.exec_()


if __name__ == '__main__':
    App = QtWidgets.QApplication(sys.argv)
    win = Main_win()
    win.show()
    sys.exit(App.exec_())