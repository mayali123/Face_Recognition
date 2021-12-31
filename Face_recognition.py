# -*- coding: UTF-8 -*-
# @Time :  21:11
# @Author :mayali123
# @File : Face_recognition.py
# @Software : PyCharm
import cv2 as cv


class Face_recognition():

    def __init__(self, svm_model=None):
        # 训练模型
        self.svm_model = svm_model


    # 对人脸进行框取 同时得到最大的矩形框的位置
    def face_detection(self, img):
        # 导入
        # haarcascade_frontalface_alt2.xml
        # haarcascade_frontalface_default.xml

        face_cascade = cv.CascadeClassifier(
            r"E:\python_project\x-ray\venv\Scripts\test.py\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml")

        # 得到人脸的 框
        faces = face_cascade.detectMultiScale(img, 1.2, 5)

        # 画矩形
        # 同时得到最大的矩形框
        max_area = 0
        return_x, return_y, return_w, return_h = 0, 0, 0, 0

        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if w * h > max_area:
                max_area = w * h
                return_x, return_y, return_w, return_h = x, y, w, h

        return img, return_x, return_y, return_w, return_h

    # 对人脸进行框取 并识别
    def face_detection_identify(self, img):
        # 导入
        face_cascade = cv.CascadeClassifier(
            r"E:\python_project\x-ray\venv\Scripts\test.py\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml")

        # 得到框出人脸的矩形
        faces = face_cascade.detectMultiScale(img, 1.2, 5)
        # 画矩形
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 识别
            selective_img = img[y:y + h + 15, x:x + w]
            # 变成黑白图
            selective_img = cv.cvtColor(selective_img, cv.COLOR_BGR2GRAY)
            # 缩小尺寸
            selective_img = cv.resize(selective_img, (92, 112))
            # 识别的结果
            prodict_id = self.svm_model.Prodict(selective_img)
            # # 显示结果
            cv.putText(img, str(prodict_id), (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv.LINE_AA)

        return img

    # 显示图片
    def cv_show(img, title='img', time=0):
        if img is None:
            print("图片不存在".format(img))
        else:
            cv.imshow(title, img)
            cv.waitKey(time)
            cv.destroyAllWindows()


# if __name__ == "__main__":
#     # main()
    