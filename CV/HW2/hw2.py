import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
def sleep(t):
    time.sleep(t)

# 1. 直方图均衡化
def histogram_equalization(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    hist, _ = np.histogram(img.flatten(), 256, [0, 255])
    #归一化hist
    hist_normalized = hist / (img.shape[0] * img.shape[1])
    cdf= hist_normalized.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # 以下代码块替代···img_equalized = cv2.equalizeHist(img)···实现直方图均衡化
    img_equalized = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
                img_equalized[i][j] = cdf[img[i][j]] * 255
    
    hist_equalized, _ = np.histogram(img_equalized.flatten(), 256, [0, 256])
    cdf_equalized = hist_equalized.cumsum()
    cdf_normalized_equalized = cdf_equalized * hist_equalized.max() / cdf_equalized.max()

    _, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 1].plot(cdf_normalized, color='b')
    axs[0, 1].plot(cdf_normalized_equalized, color='r')
    axs[0, 1].set_title('')
    axs[1, 0].imshow(img_equalized, cmap='gray')
    axs[1, 0].set_title('Equalized Image')
    axs[1, 1].hist(img.flatten(), 256, [0, 256], color='b')
    axs[1, 1].hist(img_equalized.flatten(), 256, [0, 256], color='r')
    axs[1, 1].set_title('Histogram: red line is equalized histogram')
    plt.show()

# 2. 高斯滤波和均值滤波
def gaussian_and_mean_filter(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #对原图像进行填充
    img_same = np.pad(img, ((1, 1), (1, 1)), 'constant', constant_values=0)

    # 高斯滤波
    kernel=np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    print("gaussian:\n",kernel)
    #代替img_gaussian = cv2.filter2D(img, -1, kernel)的滑动窗口滤波
    img_gaussian = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_gaussian[i][j] = np.sum(img_same[i:i+3, j:j+3] * kernel)

    # 均值滤波
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
    print("mean:\n",kernel)
    #代替img_mean = cv2.filter2D(img, -1, kernel)的滑动窗口滤波
    img_mean = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_mean[i][j] = np.sum(img_same[i:i+kernel_size, j:j+kernel_size] * kernel)

    _, axs = plt.subplots(1, 3, figsize=(15, 15))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(img_gaussian, cmap='gray')
    axs[1].set_title('Gaussian Filtered Image')
    axs[2].imshow(img_mean, cmap='gray')
    axs[2].set_title('Mean Filtered Image')
    plt.show()

# 3. Normalized Correlation
def normalized_correlation(img_path, template_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    th, tw = template.shape
    #归一化模板
    template_norm=(template-template.mean())/template.std()
    response = np.zeros((h - th + 1, w - tw + 1))
    for i in range(h - th + 1):
        for j in range(w - tw + 1):
            patch = img[i:i+th, j:j+tw]
            #归一化patch
            patch_norm=(patch-patch.mean())/patch.std()
            response[i, j] = np.sum(patch_norm * template_norm)

    response = (response - response.min()) / (response.max() - response.min())

    #找到最大值的坐标
    max_index = np.argmax(response)
    max_index = np.unravel_index(max_index, response.shape)
    print(max_index)
    #在原图中画出矩形框
    img_pred = img.copy()
    cv2.rectangle(img_pred, (max_index[1], max_index[0]), (max_index[1]+tw, max_index[0]+th), 255, 2)


    # 显示图像和响应图和结果图
    _ , axs = plt.subplots(1, 4, figsize=(15, 15))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(template, cmap='gray')
    axs[1].set_title('Template')
    axs[2].imshow(response, cmap='gray')
    axs[2].set_title('Response')
    axs[3].imshow(img_pred, cmap='gray')
    axs[3].set_title('Predicted Image')
    plt.show()


if __name__ == '__main__':
    histogram_equalization('HW2/images/lena.bmp')

    gaussian_and_mean_filter('HW2/images/lena.bmp')

    normalized_correlation('HW2/images/lena.bmp', 'HW2/images/template.bmp')

    # #获取template.png
    # img = cv2.imread('HW2/images/lena.bmp', cv2.IMREAD_GRAYSCALE)
    # h, w = img.shape
    # th, tw = h//4, w//4
    # template = img[th:th*2, tw:tw*2]
    # cv2.imwrite('HW2/images/template.bmp', template)
    # print(template.shape)
    # cv2.imshow('template', template)
    # cv2.waitKey(0)

    #参考了https://zhuanlan.zhihu.com/p/143264646