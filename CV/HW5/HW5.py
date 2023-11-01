import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# 从终端读入两个字符串代表图像名称
print("通过排列组合1-4测试图片和不同匹配算法，发现只做bf.match而不使用RANSAC算法的效果非常差劲")
img1 = input("请输入第一张图片的序号（1-4）：")
img2 = input("请输入第二张图片的序号（1-4）：")
method = input(
    "请输入匹配方法(0-3)\n (0:bf.match 1:bf.match + RANSAC 2:bf.knnMatch + RANSAC 3:FLANN + RANSAC):")

img1 = os.path.join(os.path.dirname(__file__), img1+".jpg")
img2 = os.path.join(os.path.dirname(__file__), img2+".jpg")
img1 = cv2.imread(img1)
img2 = cv2.imread(img2)

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)


if method == "1" or method == "0":
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    if len(matches) < 4:
        print("Not enough matches found.")
    else:
        good_matches = []
        good_matches = sorted(matches, key=lambda x: x.distance)

if method == "2":
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    ratio = 0.8
    for m1, m2 in matches:
        #  如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
        if m1.distance < ratio * m2.distance:
            good_matches.append([m1])

if method == "3":
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    ratio = 0.8
    for m1, m2 in matches:
        #  如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
        if m1.distance < ratio * m2.distance:
            good_matches.append([m1])


# RANSAC
if len(good_matches) > 4:
    if method == "2" or method == "3":
        src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        M, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0
        )
        good_matches = [good_matches[i]
                        for i in range(len(good_matches)) if mask[i] == 1]
    if method == "1":
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        M, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0
        )
        good_matches = [good_matches[i]
                        for i in range(len(good_matches)) if mask[i] == 1]

    if method == "0":
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        M, mask = cv2.findHomography(
            src_pts, dst_pts, 0
        )
        good_matches = [good_matches[i]
                        for i in range(len(good_matches)) if mask[i] == 1]
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    imgOut = cv2.warpPerspective(
        img2,
        M,
        (gray1.shape[1], gray1.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # 将imgOut和gray1按照透明度叠加在一起
    imgOut = cv2.addWeighted(imgOut, 0.7, img1, 0.5, 0)
else:
    print("Not enough matches found.")

if method == "2" or method == "3":
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_DEFAULT,
    )
    match_img = cv2.drawMatchesKnn(
        img1,
        kp1,
        img2,
        kp2,
        good_matches,
        None,
        **draw_params
    )

if method == "1" or method == "0":
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_DEFAULT,
    )
    match_img = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good_matches,
        None,
        **draw_params
    )

plt.subplot(231)
plt.imshow(img1)
plt.title("input1")
plt.subplot(232)
plt.imshow(img2)
plt.title("input2")
plt.subplot(233)
plt.imshow(imgOut)
plt.title("50 per cent transparency overlay each")
plt.subplot(235)
plt.imshow(match_img)
plt.title("match result")
plt.show()
