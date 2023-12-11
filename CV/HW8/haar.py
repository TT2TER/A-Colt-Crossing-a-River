import cv2
import numpy as np

# 1. 读入一幅包含人脸的照片，计算积分图
image = cv2.imread("input.jpeg")
#这个程序的参数只对input大小的人像有效，对于其他大小的人像，需要调整参数
#https://blog.csdn.net/Arctic_Beacon/article/details/84820502  cv2.imread()读取图片的坑,长宽颠倒
# print(image.shape)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Original Image", gray_image)
integral_image = cv2.integral(gray_image)#计算积分图
#将积分图归一化到0-255
# integral_image = (integral_image - np.min(integral_image)) / (np.max(integral_image) - np.min(integral_image)) * 255
# integral_image = integral_image.astype(np.uint8)
# cv2.imshow("Integral Image", integral_image)
# print(integral_image.shape)
# print(integral_image.shape[0])
# print(integral_image[integral_image.shape[0]-1][integral_image.shape[1]-1])

# 2. 使用一个上黑下白的Haar特征，基于滑动窗口的方法，在积分图上计算Haar特征的值，超过指定阈值则认为是人脸

detected_faces = []
window_size = 400
half_window_size = int(window_size / 2)
step_size = 100
threshold = 8000000
for x in range(0, integral_image.shape[0] - window_size, step_size):
    for y in range(0, integral_image.shape[1] - window_size, step_size):
        # if y>integral_image.shape[1] -2*window_size :
            # print('y',y)
        # feature_value_white = integral_image[x + window_size][y + window_size] - integral_image[x + window_size][y+half_window_size] - integral_image[x][y + window_size] + integral_image[x][y+half_window_size]
        # feature_value_black = integral_image[x+window_size][y+half_window_size] - integral_image[x+window_size][y] - integral_image[x][y+half_window_size] + integral_image[x][y]
        feature_value_white = integral_image[x + window_size][y + window_size] - integral_image[x + window_size][y] - integral_image[x+half_window_size][y + window_size] + integral_image[x+half_window_size][y]
        feature_value_black = integral_image[x+half_window_size][y+window_size] - integral_image[x+half_window_size][y] - integral_image[x][y+window_size] + integral_image[x][y]
        feature_value = +feature_value_white - feature_value_black
        # print(feature_value)
        if feature_value > threshold:
            detected_faces.append((x, y, window_size, window_size))
            print('+1')

# 3. 对检测出的结果进行非极大值抑制，得到最终的检测结果
def non_max_suppression(faces, overlap_threshold):
    if len(faces) == 0:
        return []

    faces = np.array(faces)
    x1 = faces[:, 0]
    y1 = faces[:, 1]
    x2 = faces[:, 0] + faces[:, 2]
    y2 = faces[:, 1] + faces[:, 3]

    area = faces[:, 2] * faces[:, 3]
    indices = np.argsort(y2)

    selected_faces = []
    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        selected_faces.append(faces[i])

        xx1 = np.maximum(x1[i], x1[indices[:last]])
        yy1 = np.maximum(y1[i], y1[indices[:last]])
        xx2 = np.minimum(x2[i], x2[indices[:last]])
        yy2 = np.minimum(y2[i], y2[indices[:last]])

        width = np.maximum(0, xx2 - xx1 + 1)
        height = np.maximum(0, yy2 - yy1 + 1)
        overlap = (width * height) / area[indices[:last]]

        indices = np.delete(indices, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    return selected_faces

overlap_threshold = 0.5
final_faces = non_max_suppression(detected_faces, overlap_threshold)

# 输出最终的检测结果
for (x, y, w, h) in final_faces:
    cv2.rectangle(image, (y, x), (y + w, x + h), (0, 255, 0), 2)

#将图像缩小一倍
image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))

cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
