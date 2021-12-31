# -*- coding: UTF-8 -*-
# @Time :  20:42
# @Author :mayali123
# @File : Train_svm_model.py
# @Software : PyCharm

from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score
import numpy as np
import cv2 as cv
import os
import random
import skimage.feature

class Train_svm_model():
    CUT_X = 8
    CUT_Y = 4

    def __init__(self, path):
        self.path = path

        # 得到 数据
        train_X, train_y, test_X, test_y = self.load_data()
        print("开始提取训练集特征")
        feature_train_X = self.getfeatures(train_X)
        print("开始提取测试集特征")
        feature_test_X = self.getfeatures(test_X)
        print("PCA降维")
        feature_train_X_pca, feature_test_X_pca = self.Pca(feature_train_X, feature_test_X)
        self.model = self.trainSVM(feature_train_X_pca, train_y)
        self.test(feature_test_X_pca, test_y)

    # LBP
    def LBP(self, FaceMat, R=2, P=8):
        # print(FaceMat.shape)
        LBPoperator = np.mat(np.zeros([np.shape(FaceMat)[0], np.shape(FaceMat)[1] * np.shape(FaceMat)[2]]))
        # 对每一个照片分别进行 LBP
        for i in range(np.shape(FaceMat)[0]):
            # 拉平
            LBPoperator[i, :] = skimage.feature.local_binary_pattern(FaceMat[i], P, R, method='default').flatten()
            if (i + 1) % 50 == 0:
                print("提取到", i + 1)
        # 转置
        return LBPoperator.T

    # 导入数据
    def load_data(self):
        # 训练集
        train_X = []
        train_y = []
        # 测试集
        test_X = []
        test_y = []
        # 得到当前目录下的 文件列表
        person_dirnames = os.listdir(self.path)
        for dirname in person_dirnames:
            for i in range(1, 9):
                pic_path = self.path + '/' + dirname + '/' + str(i) + '.pgm'
                # 以灰度图读入
                img = cv.imread(pic_path, 0)

                # 加入训练集
                train_X.append(img)
                train_y.append(str(dirname))
                # train_y.append(int(dirname[1:]) - 1)
            for i in range(9, 11):
                pic_path = self.path + '/' + dirname + '/' + str(i) + '.pgm'
                # 以灰度图读入
                img = cv.imread(pic_path, 0)

                # 加入测试集
                test_X.append(img)
                test_y.append(str(dirname))
                # test_y.append(int(dirname[1:]) - 1)

        # 同时打乱X和y数据集。
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(train_X)
        random.seed(randnum)
        random.shuffle(train_y)
        print("训练集大小为: {}, 测试集大小为: {}".format(len(train_X), len(test_X)))
        return np.array(train_X), np.array(train_y).T, np.array(test_X), np.array(test_y).T

    # 统计直方图
    def calHistogram(self, ImgLBPope, h_num=CUT_X, w_num=CUT_Y):
        # 112 = 14 * 8, 92 = 23 * 4
        Img = ImgLBPope.reshape(112, 92)
        H, w = np.shape(Img)
        # 把图像分为8 * 4份
        Histogram = np.mat(np.zeros((256, h_num * w_num)))
        maskx, masky = H / h_num, w / w_num
        for i in range(h_num):
            for j in range(w_num):
                # 使用掩膜opencv来获得子矩阵直方图
                mask = np.zeros(np.shape(Img), np.uint8)
                # 掩膜 是只处理有  掩膜选择的区域
                mask[int(i * maskx): int((i + 1) * maskx), int(j * masky):int((j + 1) * masky)] = 255
                hist = cv.calcHist([np.array(Img, np.uint8)], [0], mask, [256], [0, 255])

                Histogram[:, i * w_num + j] = np.mat(hist).flatten().T

        return Histogram.flatten().T

    # 提取特征
    def getfeatures(self, input_face):
        # 获得实验图像的LBP算子
        # 一列是一张图
        LBPoperator = self.LBP(input_face)

        # 获得实验图像的直方图分布
        exHistograms = np.mat(np.zeros((256 * 8 * 4, np.shape(LBPoperator)[1])))  # 行 ：256 * 8 * 4,列 ：图片数目

        for i in range(np.shape(LBPoperator)[1]):
            # 得到 一张照片的 直方图分布
            exHistogram = self.calHistogram(LBPoperator[:, i], 8, 4)
            # 将 exHistogram 加入 exHistograms中
            exHistograms[:, i] = exHistogram
        # 转置
        exHistograms = exHistograms.transpose()

        return exHistograms

    # pca 降维 默认保存150的属性
    def Pca(self, train_X, test_X, n_components=150):
        # randomized ：采用Halko等的方法进行随机化奇异值分解。
        self.pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
        self.pca.fit(train_X)
        train_X_pca = self.pca.transform(train_X)
        test_X_pca = self.pca.transform(test_X)

        return train_X_pca, test_X_pca

    # 训练SVM模性
    def trainSVM(self, x_train, y_train):
        # SVM生成和训练
        # probability为true 启动使用5倍交叉验证
        clf = svm.SVC(kernel='rbf', probability=True)
        clf.fit(x_train, y_train)
        return clf

    # 测试模型
    def test(self, x_test, y_test):
        # 预测结果
        y_pre = self.model.predict(x_test)

        # 混淆矩阵
        con_matrix = confusion_matrix(y_test, y_pre)
        print('confusion_matrix:\n', con_matrix)
        print('accuracy:{}'.format(accuracy_score(y_test, y_pre)))
        print('precision:{}'.format(precision_score(y_test, y_pre, average='micro')))
        print('recall:{}'.format(recall_score(y_test, y_pre, average='micro')))
        print('f1-score:{}'.format(f1_score(y_test, y_pre, average='micro')))



    # 识别人脸 并放回下标
    def Prodict(self, img):
        # 提取特征
        img = self.getfeatures(np.array([img]))
        img = self.pca.transform(img)
        return self.model.predict(img)[0]


if __name__ == '__main__':
    svm_model = Train_svm_model('./orl')
    # svm_model.get_model()
    # svm_modelProdict