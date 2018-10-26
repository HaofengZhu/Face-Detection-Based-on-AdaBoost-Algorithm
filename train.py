from feature import *
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from ensemble import AdaBoostClassifier
if __name__ == "__main__":
    # write your code here
    x=[]
    y=[]
    for i in tqdm(range(500)):
        img = cv2.imread("./datasets/original/face/face_" +str(i).rjust(3,'0')+".jpg")
        img=cv2.resize(img,dsize=(24,24))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
        npd = NPDFeature(img)
        x.append(npd.extract())
        y.append(1)
    for i in tqdm(range(500)):
        img = cv2.imread("./datasets/original/nonface/nonface_" +str(i).rjust(3,'0')+".jpg")
        img = cv2.resize(img, dsize=(24, 24))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
        npd = NPDFeature(img)
        x.append(npd.extract())
        y.append(-1)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    print('begin train data')
    ada = AdaBoostClassifier()
    ada.fit(x_train, y_train)
    y_predict = ada.predict(x_val, threshold=0)
    print(classification_report(y_val, y_predict,
                                target_names=["nonface", "face"], digits=4))
    with open("report.txt", "w") as f:
        f.write(classification_report(y_val, y_predict,
                                target_names=["nonface", "face"], digits=4))