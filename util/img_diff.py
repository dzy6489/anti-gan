import cv2
print('VX公众号: 桔子code / juzicode.com')
print('cv2.__version__:',cv2.__version__)


def image_binarization(img):
    # 将图片转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # retval, dst = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    # 最大类间方差法(大津算法)，thresh会被忽略，自动计算一个阈值
    retval, dst = cv2.threshold(gray, 149, 255, cv2.THRESH_BINARY)
    cv2.imshow('binary.jpg', dst)
    cv2.imwrite('binary.jpg',dst)
img_name='1023'
img = cv2.imread('./images/tb'+img_name+'_fake.png')
img2 = cv2.imread('./images/tb'+img_name+'_real.png' )
img3 = cv2.subtract(img2,img)
cv2.imshow('subtract(img,img2)',img3)
cv2.imwrite('sub.png',img3)
img4 = cv2.imread('./images/tb'+img_name+'_fake.png')
image_binarization(img4)
cv2.imshow('im4',img4)

# img3 = cv2.subtract(img2,img)
# print('img2[161,199]:',img2[161,199])
# print('img[161,199]:',img[161,199])
# print('img3[161,199]:',img3[161,199])
# cv2.imshow('subtract(img2,img)',img3)

cv2.waitKey(0)