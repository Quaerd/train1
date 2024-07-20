import cv2
import numpy as np
import math

# 纸币的长宽
w=156
h=76
# 蓝色
lower_blue = np.array ([100, 90, 50])
upper_blue = np.array ([130,255, 255])
# 绿色
lower_green = np.array ([35, 50, 50])
upper_green = np.array ([90, 255, 255])

# cv2.namedWindow('img',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('img',1024,800)
# cv2.namedWindow('ini',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('ini',800,600)

img=cv2.imread('D:/learn/train_2024/train1/testpic/10.jpg')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
img_copy=img.copy()
img_blur=cv2.GaussianBlur(img_gray,(5,5),0)
thresh=cv2.threshold(img_blur,140,255,cv2.THRESH_BINARY)[1]
opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8)
# background = np.ones_like(img_gray) * 255
# for i in range(1, num_labels):
#     mask = np.uint8(labels == i) * 255
#     background = cv2.bitwise_and(background, background, mask=mask)
#     background = cv2.dilate(background, kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=1)
# contours= cv2.findContours(background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
# for contour in contours:
#     peri = cv2.arcLength(contour, True)
#     epsilon = 0.02 * peri
#     approx = cv2.approxPolyDP(contour, epsilon, True)
#     print("detect")
#     if len(approx) == 4:
#         cv2.drawContours(img_copy, [approx], -1, (0, 255, 0), 2)
#         points = approx.reshape(4, 2)
#         print("矩形的顶点:", points)
# cv2.imshow('Detected Rectangles', img_copy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

contour=cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
# img_copy=img.copy()
# cv2.drawContours(img_copy,contour,-1,(0,0,255),3)
# cv2.imshow('img',img_copy)
# 透视变换矩阵
money_contour=sorted(contour,key=cv2.contourArea,reverse=True)[0]
epsilon=0.05*cv2.arcLength(money_contour,True)
approx=cv2.approxPolyDP(money_contour,epsilon,True)
points_array = np.array([point[0] for point in approx],dtype=np.float32)
points_array=sorted(points_array,key=lambda point:point[0]+point[1])
if len(points_array)!=4:
    print("None rectangle")
a=points_array[1]
b=points_array[2]
points_array[2]=points_array[3]
points_array[1]=b
points_array[3]=a
dst_array=np.float32([[0,0],[w,0],[w,h],[0,h]])
dis1=math.sqrt((points_array[1][0]-points_array[0][0])**2+(points_array[1][1]-points_array[0][1])**2)
dis2=math.sqrt((points_array[3][0]-points_array[0][0])**2+(points_array[3][1]-points_array[0][1])**2)
if(dis2>dis1):
    dst_array[1]=[h,0]
    dst_array[3]=[0,w]
points_array=np.array(points_array,dtype=np.float32)
print(points_array)
# points_array=points_array.astype(np.float32)
M=cv2.getPerspectiveTransform(points_array,dst_array)
dst=cv2.warpPerspective(img_copy,M,(w,h))
cv2.imshow('dst',dst)
# 处理正方形
mask_g=cv2.inRange(img_hsv,lower_green,upper_green)
# bitwise_g=cv2.bitwise_and(img_hsv,img_hsv,mask=mask_g)
# cv2.imshow('img',bitwise_g)
# bitwise_g_gray=cv2.cvtColor(bitwise_g,cv2.COLOR_BGR2GRAY)
rect_contours=cv2.findContours(mask_g,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
rect_contour=sorted(rect_contours,key=cv2.contourArea,reverse=True)[0]
rect_epsilon=0.02*cv2.arcLength(rect_contour,True)
rect_approx=cv2.approxPolyDP(rect_contour,rect_epsilon,True)
rect_array=np.array([point[0] for point in rect_approx],dtype=np.float32)
rect_array=sorted(rect_array,key=lambda point:point[0]+point[1])
a=rect_array[1]
b=rect_array[2]
rect_array[2]=rect_array[3]
if a[0]>b[0]:
    rect_array[1]=a
    rect_array[3]=b
elif b[0]>a[0]:
    rect_array[1]=b
    rect_array[3]=a
rect_array=np.array(rect_array,dtype=np.float32)

print(rect_array)
#cv2.imshow('dst',img)
# 将点扩展为齐次坐标
ones = np.ones((rect_array.shape[0], 1))
points_homogeneous = np.hstack([rect_array, ones])
transformed_points = M @ points_homogeneous.T
transformed_points /= transformed_points[2]  # 归一化
# 提取转换后的顶点坐标
t = transformed_points[:2].T
# print(t)
l1=math.sqrt((t[1][0]-t[0][0])**2+(t[1][1]-t[0][1])**2)
l2=math.sqrt((t[2][0]-t[1][0])**2+(t[2][1]-t[1][1])**2)
l3=math.sqrt((t[3][0]-t[2][0])**2+(t[3][1]-t[2][1])**2)
l4=math.sqrt((t[0][0]-t[3][0])**2+(t[0][1]-t[3][1])**2)
l=(l1+l2+l3+l4)/4
print(l)

# 处理圆
img_hsv=cv2.GaussianBlur(img_hsv,(5,5),0)
mask_b=cv2.inRange(img_hsv,lower_blue,upper_blue)
# cv2.imshow('mask',mask_b)
# cv2.imshow('ini',img_hsv)
# bitwise_b=cv2.bitwise_and(img_hsv,img_hsv,mask=mask_b)
# cv2.imshow('blue',bitwise_b)
# cv2.imshow('ini',mask_b)
mask_b=cv2.threshold(mask_b,127,255,cv2.THRESH_BINARY)[1]
circle_opening=cv2.morphologyEx(mask_b,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
# cv2.imshow('mask',circle_opening)
contours=cv2.findContours(mask_b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
contour=sorted(contours,key=cv2.contourArea,reverse=True)[0]
# hsv_copy=img_hsv.copy()
# cv2.drawContours(hsv_copy,contour,-1,(0,0,255),3)
# cv2.imshow('hsv',hsv_copy)
# ellipse=cv2.fitEllipse(contour)
contour_homogeneous = cv2.convertPointsToHomogeneous(contour)[:, 0, :]
original_contour_homogeneous = np.dot(M, contour_homogeneous.T).T
original_contour_homogeneous /= original_contour_homogeneous[:, -1:]
original_contour = original_contour_homogeneous[:, :2]
(x, y), radius = cv2.minEnclosingCircle(original_contour.astype(np.float32))
dia=radius*2
print(dia)






# 透视变换
# dst=cv2.warpPerspective(img,M,(w,h))
# print(points_array)
# cv2.imshow('imga',dst)

# draw=np.zeros(img.shape,np.uint8)
# cv2.drawContours(draw,[money_contour],0,255,1)
# draw_gray=cv2.cvtColor(draw,cv2.COLOR_BGR2GRAY)
# dst=cv2.cornerHarris(draw_gray,2,3,0.04)
# dst=cv2.dilate(dst,None)
# corners=np.argwhere(dst>dst.max()*0.01)
# print(len(corners))


#)
# cv2.imshow('img',opening)
# cv2.imshow('ini',img)
# cv2.imshow('i',img)
#cv2.imshow('img',img_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()