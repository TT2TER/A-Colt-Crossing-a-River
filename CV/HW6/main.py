import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from mycnn import CNN
import torch.nn as nn
import os

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
    
BATCH_SIZE = 64

is_test = 1

if __name__ == "__main__":
    seed=0
    torch.manual_seed(0)  # cpu
    torch.cuda.manual_seed(0)  # GPU
    # 设置 Python 的哈希种子。
    # 在某些情况下，Python 的哈希值可以影响一些数据结构的顺序，通过设置这个种子，可以确保在不同运行中哈希值的计算是一致的。
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 控制 PyTorch 中的 cuDNN 库的行为
    # 此选项确保 cuDNN 使用确定性算法执行操作。这意味着相同的操作在不同运行中将生成相同的结果。
    # 这对于实现训练的可重现性非常重要，因为在深度学习中，有些优化可能涉及到随机性操作，例如权重初始化
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
    # print(training_data.data.shape)
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    print('可以通过改变main.py中的全局变量is_test来选择训练或测试')
    if not is_test:
        fasioncnn = CNN()
        DEVICE = torch.device("cpu")
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        print(DEVICE)
        fasioncnn = fasioncnn.to(DEVICE)

        criterion = nn.CrossEntropyLoss().to(DEVICE)

        #用亞當優化器
        LEARNING_RATE = 0.01
        optimizer = torch.optim.Adam(fasioncnn.parameters(), lr=LEARNING_RATE)

        TOTAL_EPOCHS = 5
        losses = []
        for epoch in range(TOTAL_EPOCHS):
            #在每个批次下,遍历每个训练样本
            for i, (images, labels) in enumerate(train_dataloader):
                images = images.float().to(DEVICE)
                #将标签转换独热向量
                labels = nn.functional.one_hot(labels).float()
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = fasioncnn(images)
                # print(labels)
                #计算损失函数
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                losses.append(loss.to('cpu').data.item())
                if (i+1) % 100 == 0:
                    print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'%(epoch+1, TOTAL_EPOCHS, i+1, len(training_data)//BATCH_SIZE, loss.data.item()))

        plt.xlabel('Epoch #')
        plt.ylabel('Loss')
        plt.plot(losses)
        plt.show()

        torch.save(fasioncnn.state_dict(), 'model_parameters.pth')

    if is_test:
        print('开始测试……\n')
        fasioncnn = CNN()
        DEVICE = torch.device("cpu")
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        print(DEVICE)
        fasioncnn = fasioncnn.to(DEVICE)
        fasioncnn.load_state_dict(torch.load('model_parameters.pth'))
        fasioncnn.eval()
        correct = 0
        total = 0
        for images, labels in test_dataloader:
            images = images.float().to(DEVICE)
            outputs = fasioncnn(images).to('cpu')
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            #每100個圖片展示一張圖片和對應預測類別和實際類別
            # if total % 100 == 0:
            #     plt.imshow(images[0].cpu().squeeze().numpy())
            #     plt.show()
            #     print('Predicted: ', ' '.join('%5s' % predicted[j] for j in range(1)))
            #     print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(1)))
        print('测试准确率: %.4f %%' % (100 * correct / total))


