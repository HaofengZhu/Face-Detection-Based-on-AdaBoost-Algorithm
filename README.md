# 机器学习 AdaBoost与人脸检测
##简介
使用 NPDFeature提取图像特征并使用AdaBoost进行二分类任务，判断该图片是否是人脸
其中AdaBoost使用sklearn的DecisionTreeClassifier作为基分类器

##文档说明
datasets:存储了500张人脸图和500张非人脸图，用于测试
face_detection:是一个用于标注人脸位置的库
ensemble.py:AdaBoost类源码
feature.py：NPDFeature类源码
train.py：AdaBoost测试，读取测试集并训练，测试准确率

##准确率
使用sklearn的classification_report计算precision和recall，并将结果存储进report.txt