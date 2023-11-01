import cv2
import matplotlib.pyplot as plt
import numpy as np

# 在这里bin的数量作为参数可调
bin_num = 9


def interpolate_bins(bin_index, angle, bin_width):
    # 计算相邻的两个中心的索引
    if angle == 180:
        angle = 0
    bin1 = (bin_index + bin_num) % bin_num
    bin2 = (bin1 + 1) % bin_num

    # 计算角度差值
    diff1 = (angle - bin1 * bin_width) / bin_width
    diff2 = 1 - diff1

    return bin1, bin2, diff1, diff2


if __name__ == "__main__":
    img = cv2.imread("./zhou.jpg")

    # 将图片分四块
    height, width = img.shape[:2]
    block_height = height // 2
    block_width = width // 2

    blocks = []
    for i in range(2):
        for j in range(2):
            block = img[
                i * block_height : (i + 1) * block_height,
                j * block_width : (j + 1) * block_width,
            ]
            blocks.append(block)

    # 分block计算梯度方向直方图

    histograms = []
    for block in blocks:
        gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        # print(magnitude.shape)
        # time.sleep(5)
        direction = np.arctan2(sobely, sobelx) * 180 / np.pi
        direction[direction < 0] += 180
        # #画四幅图分别显示sobelx, sobely, magnitude, direction
        # plt.subplot(221)
        # plt.imshow(sobelx, cmap='gray')
        # plt.title('sobelx')
        # plt.subplot(222)
        # plt.imshow(sobely, cmap='gray')
        # plt.title('sobely')
        # plt.subplot(223)
        # plt.imshow(magnitude, cmap='gray')
        # plt.title('magnitude')
        # plt.subplot(224)
        # plt.imshow(direction, cmap='gray')
        # plt.title('direction')
        # plt.show()
        histogram = np.zeros(bin_num)
        bin_width = 180 / bin_num
        for i in range(magnitude.shape[0]):
            for j in range(magnitude.shape[1]):
                # 将角度按照比例分配给对应的bin
                # 角度的范围是[0,180)
                # 为了方便计算，将0,0+bin_width,0+2*bin_width,...作为每个bin的中心,0为中心的bin为第0个bin
                # bin的范围为(180-bin_width/2,180)∩[0,bin_width/2),[0+bin_width/2,0+bin_width+bin_width/2),...
                # 而不是课上讲的0+bin_width/2,0+3*bin_width/2,...作为中心
                # 这样计算的话，0,bin_width,2*bin_width,...会被直接分配到对应索引的bin中
                # 而在bin范围内的会根据距离两个中心的距离进行按比例分配
                bin_index = (
                    int(direction[i, j] // bin_width) % bin_num
                )  # 找到起始bin索引
                bin1, bin2, diff1, diff2 = interpolate_bins(
                    bin_index, direction[i, j], bin_width
                )  # 找到相邻的两个bin的索引和角度差值
                # print(direction[i,j])
                # print(bin1, bin2, diff1, diff2)
                # time.sleep(1)
                histogram[bin1] += magnitude[i, j] * diff2  # 按比例分配
                histogram[bin2] += magnitude[i, j] * diff1  # 按比例分配
                if i % 1000 == 0 and j % 1000 == 0:
                    # 这一步是采样，展示说明过程的正确性
                    print("sample show:")
                    print("angle", direction[i, j], "mag", magnitude[i, j])
                    print(
                        "bin", bin1 * bin_width, "degree, add", magnitude[i, j] * diff2
                    )
                    print(
                        "bin", bin2 * bin_width, "degree, add", magnitude[i, j] * diff1
                    )

        histograms.append(histogram)

    # 可视化
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(cv2.cvtColor(blocks[0], cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Block 1")
    axs[0, 1].imshow(cv2.cvtColor(blocks[1], cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title("Block 2")
    axs[1, 0].imshow(cv2.cvtColor(blocks[2], cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title("Block 3")
    axs[1, 1].imshow(cv2.cvtColor(blocks[3], cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title("Block 4")

    fig2, axs2 = plt.subplots(2, 2)
    axs2[0, 0].bar(np.arange(bin_num), histograms[0])
    axs2[0, 0].set_title("Histogram for Block 1")
    axs2[0, 1].bar(np.arange(bin_num), histograms[1])
    axs2[0, 1].set_title("Histogram for Block 2")
    axs2[1, 0].bar(np.arange(bin_num), histograms[2])
    axs2[1, 0].set_title("Histogram for Block 3")
    axs2[1, 1].bar(np.arange(bin_num), histograms[3])
    axs2[1, 1].set_title("Histogram for Block 4")

    plt.show()
