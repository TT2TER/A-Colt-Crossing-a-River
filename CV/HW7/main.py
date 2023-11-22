import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
# from imutils import paths
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
def filter_labels(data):
    # 获取数据和标签
    data, labels = data
    # 只保留标签在前5个类别中的样本
    mask = labels < 5
    return data[mask], labels[mask]

def visualize_kmeans(kmeans, sift_features):
    # Reduce dimensionality for visualization (using PCA)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(sift_features)
    reduced_centers = pca.transform(kmeans.cluster_centers_)

    # Plot the k-means clusters
    plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=kmeans.labels_,
        cmap="viridis",
        s=5,
        alpha=0.5,
    )
    plt.scatter(
        reduced_centers[:, 0],
        reduced_centers[:, 1],
        c="red",
        marker="X",
        s=100,
        label="Cluster Centers",
    )
    plt.title("K-Means Clustering")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

BATCH_SIZE = 1

test = 1

if __name__ == "__main__":
    seed=0
    torch.manual_seed(0) 
    torch.cuda.manual_seed(0)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    #選取前五個類別
    # print(training_data.data.shape)
    training_data.data, training_data.targets = filter_labels((training_data.data, training_data.targets))
    test_data.data, test_data.targets = filter_labels((test_data.data, test_data.targets))
    train_images, train_labels = training_data.data, training_data.targets
    test_images, test_labels = test_data.data, test_data.targets

    print('可以通过改变main.py中的全局变量is_test来选择训练或测试')
    if not test:
        sifts_img = [] # 存放所有图像的文件名和sift特征

        limit = 30000 #最大训练个数

        count = 0 # 词袋特征个数
        num = 0 # 有效个数
        label = []
        for i in range(limit):
            # if i%3 == 0:
                img = train_images[i]
                # print(img)
                img = np.uint8(img)
                # print(img)
                sift = cv2.xfeatures2d.SIFT_create()
                kp,des = sift.detectAndCompute(img,None)#des是描述子
                if des is None:#没有检测到特征点，des为None
                    continue
                sifts_img.append(des)
                label.append(train_labels[i])
                # print(des)
                # print(des.shape[0])
                count = count + des.shape[0]
                num = num + 1
                print(i,'/',limit)
            # else:
                # continue

        label = np.array(label)

        data = sifts_img[0]
        for des in sifts_img[1:]:
            # print('1',des.shape)
            data = np.vstack((data, des))#纵向合并矩阵,比如两个矩阵都是2行3列的，合并后就是4行3列的
            # print(data.shape)

        print("train file:",num)
        count = int(count / 40)
        count = max(4,count)
        # 对sift特征进行聚类
        k_means = KMeans(n_clusters=int(count), n_init=4)
        k_means.fit(data)

        visualize_kmeans(k_means, data)


        # 构建所有样本的词袋表示
        image_features = np.zeros([int(num),int(count)],'float32')
        for i in range(int(num)):
            ws, d = vq(sifts_img[i],k_means.cluster_centers_)# 计算各个sift特征所属的视觉词汇
            for w in ws:
                image_features[i][w] += 1  # 对应视觉词汇位置元素加1


        x_tra, x_val, y_tra, y_val = train_test_split(image_features,label,test_size=0.2)
        # 构建线性SVM对象并训练
        clf = LinearSVC(C=1, loss="hinge").fit(x_tra, y_tra)
        # 训练数据预测正确率
        print (clf.score(x_val, y_val))




        # save the training model as pickle
        with open('bow_kmeans.pickle','wb') as fw:
            pickle.dump(k_means,fw)
        with open('bow_clf.pickle','wb') as fw:
            pickle.dump(clf,fw)
        with open('bow_count.pickle','wb') as fw:
            pickle.dump(count,fw)
        print('Trainning successfully and save the model')

    if test:
        with open('bow_kmeans.pickle','rb') as fr:
            k_means = pickle.load(fr)
        with open('bow_clf.pickle','rb') as fr:
            clf = pickle.load(fr)
        with open('bow_count.pickle','rb') as fr:
            count = pickle.load(fr)

        target_file = ['T-shirt','Trouser','Pullover',
                    'Dress','Coat']

        plt.figure()
        cnt = 30
        i = 1
        while(i<=12):
            img = test_images[cnt]
            cnt = cnt + 1
            img = np.uint8(img)
            sift = cv2.xfeatures2d.SIFT_create()
            kp,des = sift.detectAndCompute(img,None)
            if des is None:
                continue
            words, distance = vq(des, k_means.cluster_centers_)
            image_features_search = np.zeros((int(count)), "float32")
            for w in words:
                image_features_search[w] += 1
            t = clf.predict(image_features_search.reshape(1,-1))
            plt.subplot(3,4,i)
            i += 1
            plt.imshow(img,'gray')
            plt.title(target_file[t[0]])
            plt.axis('off')
        plt.show()

        i = 0
        len = test_images.shape[0]
        predict_arr = []

        while(i<len):
            img = test_images[i]
            img = np.uint8(img)
            sift = cv2.xfeatures2d.SIFT_create()
            # print(i)
            kp,des = sift.detectAndCompute(img,None)
            if des is None:
                i += 1
                predict_arr.append(0)
                continue
            words, distance = vq(des, k_means.cluster_centers_)
            image_features_search = np.zeros((int(count)), "float32")
            for w in words:
                image_features_search[w] += 1
            t = clf.predict(image_features_search.reshape(1,-1))
            i += 1
            predict_arr.append(t[0])
        score=accuracy_score(test_labels,predict_arr)
        print(score)
        print(classification_report(test_labels,predict_arr))
