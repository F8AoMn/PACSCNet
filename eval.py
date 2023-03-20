import numpy as np
import os
from skimage import io
from dataprocessing import *
__all__ = ['SegmentationMetric']

# DATASET = "Vaihingen"
DATASET = "Potsdam"
if DATASET == "Vaihingen":
    index = 5
elif DATASET == "Potsdam":
    index = 6
"""

confusionMetric
P\L     P    N

P      TP    FP

N      FN    TN

"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2) # onfusionMatrix n*n , The initial values are all 0
    # Pixel accuracy PA, predicting correct pixels / total pixels

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        # self.confusionMatrix = self.confusionMatrix[0:5, 0:5]
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc
    # Class pixel accuracy CPA, returns the value of n * 1, representing each category, including background

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precison)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    # Class average pixel accuracy MPA, the pixel accuracy of each class is averaged
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        # print("Imp.surf:{:.2f}%\nBuildings:{:.2f}%\nLow veg.:{:.2f}%\nTree:{:.2f}%\nCar:{:.2f}%\n".format(classAcc[0]*100,classAcc[1]*100,classAcc[2]*100,classAcc[3]*100,classAcc[4]*100))
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        return IoU

    # MIoU
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def F1Score(self):
        precision = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        recall = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        f1score = 2 * precision * recall / (precision + recall)
        return f1score

    # According to the label and prediction image, the confusion matrix is returned
    def genConfusionMatrix(self, imgPredit, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredit[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        # confusionMatrix = confusionMatrix[0:5, 0:5]
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =   [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
            np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
            np.diag(self.confusionMatrix))
        FWIOU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIOU

    # Update confusion matrix
    def addBatch(self, imgPredict, imgLabel):
        print('imgpredict.shape: {}\nimgLabel.shape:{}'.format(imgPredict.shape, imgLabel.shape))
        assert imgPredict.shape == imgLabel.shape  # Make sure that the size of the tag and prediction image is equal
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    # Clear confusion matrix
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


def evaluate1(pre_path, label_path):
    acc_list = []
    macc_list = []
    mIoU_list = []
    fwIoU_list = []
    if DATASET == 'Vaihingen':
        IoU_list = [[], [], [], [], []]
        classacc_list = [[], [], [], [], []]  # Vaihingen
    elif DATASET == 'Potsdam':
        IoU_list = [[], [], [], [], [], []]
        classacc_list = [[], [], [], [], [], []]  # Potsdam
    f1score_list = []

    pre_imgs = os.listdir(pre_path)
    pre_imgs.sort()
    lab_imgs = os.listdir(label_path)
    lab_imgs.sort()
    for i, p in enumerate(pre_imgs):
        print(i, p)
        print()
        imgPredict = convert_from_color(io.imread(pre_path + p))
        imgPredict = np.array(imgPredict)
        # imgPredict = imgPredict[:,:,0]
        imgLabel = convert_from_color(io.imread(label_path+lab_imgs[i]))
        imgLabel = np.array(imgLabel)
        # imgLabel = imgLabel[:,:,0]

        metric = SegmentationMetric(index)  # Represents the number of categories, including background
        metric.addBatch(imgPredict, imgLabel)
        acc = metric.pixelAccuracy()
        # print("Imp.surf:{:.2f}%\nBuildings:{:.2f}%\nLow veg.:{:.2f}%\nTree:{:.2f}%\nCar:{:.2f}%\n".format(classAcc[0]*100,classAcc[1]*100,classAcc[2]*100,classAcc[3]*100,classAcc[4]*100))
        classacc = metric.classPixelAccuracy()
        IoU = metric.IntersectionOverUnion()
        macc = metric.meanPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        fwIoU = metric.Frequency_Weighted_Intersection_over_Union()
        f1score = metric.F1Score()
        f1score1 = []
        for i in range(index):
            f1score1.append( f1score[i] )
        print("F1Score:", f1score)
        print("F1Score1:", f1score1)

        acc_list.append(acc)
        macc_list.append(macc)
        mIoU_list.append(mIoU)
        fwIoU_list.append(fwIoU)
        f1score_list.append(np.nanmean(f1score1))
        for i in range(index):
            classacc_list[i].append(classacc[i])
        for i in range(index):
            IoU_list[i].append(IoU[i])

        # print('{}: acc={}, macc={}, mIoU={}, fwIou={}'.format(p, acc, macc, mIoU, fwIoU))

    return acc_list, macc_list, mIoU_list, fwIoU_list, classacc_list, f1score_list, IoU_list


def evaluate2(pre_path, label_path):
    pre_imgs = os.listdir(pre_path)
    lab_imgs = os.listdir(label_path)

    metric = SegmentationMetric(index) # Represents the number of categories, including background
    for i, p in enumerate(pre_imgs):
        imgPredict = io.imread(pre_path+p)
        imgPredict = np.array(imgPredict)


if __name__ == '__main__':
    pre_path = './results/{}/PACSCNet/'.format(DATASET)
    label_path = '/home/user/Disk/JJH/ISPRS/test/{}/'.format(DATASET)

    acc_list, macc_list, mIoU_list, fwIoU_list, classAcc, f1score_list, IoU_list = evaluate1(pre_path, label_path)
    print('final1: acc={:.2f}%, F1-socre={:.2f}, macc={:.2f}%,mIoU={:.2f}%,fwIoU={:.2f}%'.format(
        np.mean(acc_list)*100, np.mean(f1score_list)*100, np.mean(macc_list)*100,
        np.mean(mIoU_list)*100, np.mean(fwIoU_list)*100))
    if DATASET == 'Vaihingen':
        print("Imp.surf:{:.2f}%\nBuildings:{:.2f}%\nLow veg.:{:.2f}%\nTree:{:.2f}%\nCar:{:.2f}%".format(
            np.mean(classAcc[0]) * 100, np.mean(classAcc[1]) * 100, np.mean(classAcc[2]) * 100,
            np.mean(classAcc[3]) * 100, np.mean(classAcc[4]) * 100))
        print("Imp.surf IoU:{:.2f}%\nBuildings IoU:{:.2f}%\nLow veg. IoU:{:.2f}%\nTree IoU:{:.2f}%\nCar IoU:{:.2f}%".format(
                np.mean(IoU_list[0]) * 100, np.mean(IoU_list[1]) * 100, np.mean(IoU_list[2]) * 100,
                np.mean(IoU_list[3]) * 100, np.mean(IoU_list[4]) * 100))  # Vaihingen
    elif DATASET == 'Potsdam':
        print("Imp.surf:{:.2f}%\nBuildings:{:.2f}%\nLow veg.:{:.2f}%\nTree:{:.2f}%\nCar:{:.2f}%\nClutter:{:.2f}%".format(
            np.mean(classAcc[0])*100,np.mean(classAcc[1])*100,np.mean(classAcc[2])*100,np.mean(classAcc[3])*100,
            np.mean(classAcc[4])*100, np.mean(classAcc[5])*100))
        print("Imp.surf IoU:{:.2f}%\nBuildings IoU:{:.2f}%\nLow veg. IoU:{:.2f}%\nTree IoU:{:.2f}%\nCar IoU:{:.2f}%\nClutter IoU:{:.2f}%".format(
                np.mean(IoU_list[0])*100, np.mean(IoU_list[1])*100, np.mean(IoU_list[2])*100,
                np.mean(IoU_list[3])*100, np.mean(IoU_list[4])*100, np.mean(IoU_list[5])*100))
