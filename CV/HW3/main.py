import cv2
import numpy as np
import matplotlib.pyplot as plt

INF = 1e9

# 1. 载入图像，进行傅里叶变换，显示得到的频谱图像
def plot_spectrum(image):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 进行傅里叶变换
    f_transform = np.fft.fft2(gray_image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # 计算幅度谱
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
    
    # 显示频谱图像
    plt.subplot(121), plt.imshow(gray_image, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

# 2. 对得到的频谱图，去除高频部分，进行反变换，显示得到的图像
def remove_high_frequencies(image, cutoff_ratio=0.2):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 进行傅里叶变换
    f_transform = np.fft.fft2(gray_image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # 获取频谱图像的中心坐标
    rows, cols = gray_image.shape
    center_row, center_col = rows // 2, cols // 2
    
    # 设置高频部分的中心区域为零
    cutoff_frequency = int(cutoff_ratio * min(center_row, center_col))
    f_transform_shifted[:center_row - cutoff_frequency] = 0
    f_transform_shifted[center_row + cutoff_frequency:] = 0
    f_transform_shifted[:, :center_col - cutoff_frequency] = 0
    f_transform_shifted[:, center_col + cutoff_frequency:] = 0
    
    
    # 计算幅度谱
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)

    # 进行反傅里叶变换
    f_transform_inverse = np.fft.ifftshift(f_transform_shifted)
    image_restored = np.fft.ifft2(f_transform_inverse)
    image_restored = np.abs(image_restored).astype(np.uint8)
    
    # 显示去除高频部分后的图像
    plt.subplot(121), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('magnitude_spectrum with High Frequencies Removed'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(image_restored, cmap='gray')
    plt.title('Image with High Frequencies Removed'), plt.xticks([]), plt.yticks([])
    plt.show()

# 3. 对得到的频谱图，去除低频部分，进行反变换，显示得到的图像
def remove_low_frequencies(image, cutoff_ratio=0.2):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 进行傅里叶变换
    f_transform = np.fft.fft2(gray_image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # 获取频谱图像的中心坐标
    rows, cols = gray_image.shape
    center_row, center_col = rows // 2, cols // 2
    
    # 设置低频部分的中心区域为零
    cutoff_frequency = int(cutoff_ratio * min(center_row, center_col))
    f_transform_shifted[center_row - cutoff_frequency:center_row + cutoff_frequency,
                        center_col - cutoff_frequency:center_col + cutoff_frequency] = INF
    
    # 计算幅度谱
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
    # 进行反傅里叶变换
    f_transform_inverse = np.fft.ifftshift(f_transform_shifted)
    image_restored = np.fft.ifft2(f_transform_inverse)
    image_restored = np.abs(image_restored).astype(np.uint8)
    
    # 显示去除低频部分后的图像
    plt.subplot(121), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('magnitude_spectrum with Low Frequencies Removed'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(image_restored, cmap='gray')
    plt.title('Image with Low Frequencies Removed'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    # 读取图像
    image = cv2.imread('HW3/images/lena.bmp')
    
    # 1. 显示频谱图
    plot_spectrum(image)
    
    # 2. 去除高频部分，显示结果
    remove_high_frequencies(image, cutoff_ratio=0.2)
    
    # 3. 去除低频部分，显示结果
    remove_low_frequencies(image, cutoff_ratio=0.2)
