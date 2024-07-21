import math
import os
import cv2
import numpy as np

def money_area(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_copy=img.copy()

    # color
    lower_red2 = np.array([170, 50, 100])
    upper_red2 = np.array([180, 55, 255])
    lower_w=np.array([0,0,200])
    upper_w=np.array([200,255,255])
    # blue
    lower_blue = np.array ([100, 90, 50])
    upper_blue = np.array ([130,255, 255])
    # green
    lower_green = np.array ([35, 50, 50])
    upper_green = np.array ([90, 255, 255])
    mask_r = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_w=cv2.inRange(hsv, lower_w, upper_w)
    mask_b=cv2.inRange(hsv, lower_blue, upper_blue)
    mask_g=cv2.inRange(hsv, lower_green, upper_green)
    mask_white_light_red = cv2.bitwise_or(mask_w, mask_r)

    # 排除绿色和蓝色的区域
    mask_excluded = cv2.bitwise_or(mask_g, mask_b)
    mask = cv2.bitwise_and(mask_white_light_red, cv2.bitwise_not(mask_excluded))
    kernel = np.ones((5,5),np.uint8)
    # mask=cv2.dilate(mask,kernel,iterations=2)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=20)
    mask1=cv2.bitwise_and(img_copy, img_copy, mask=mask)
    # cv2.imshow('mask',mask)

    img1=img.copy()
    contours= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(img1, contours, -1, (0, 255, 0), 2)
    contours_sort=sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # cv2.drawContours(img1, contours_sort, -1, (0, 255, 0), 2)
    # cv2.imshow('imga',img1)

    peri=cv2.arcLength(contours_sort, True)
    approx=cv2.approxPolyDP(contours_sort,peri*0.015,True)
    hull = cv2.convexHull(approx)
    # cv2.imshow('contours',img_copy)
    # print(approx)
    area=cv2.contourArea(hull)
    return area
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def square_area(img):
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('img',800,600)
    # green
    lower_green = np.array ([40, 100, 100])
    upper_green = np.array ([85, 255, 255])
    mask_g=cv2.inRange(img_hsv,lower_green,upper_green)
    img=cv2.bitwise_and(img,img,mask=mask_g)
    mask=cv2.morphologyEx(mask_g,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8),iterations=2)
    contour=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    contour_sort=sorted(contour,key=cv2.contourArea,reverse=True)[0]
    # cv2.drawContours(img,contour,0,(0,0,255),3)
    point=np.array(contour_sort,np.int32)
    point=point.reshape((-1,1,2))
    area=cv2.contourArea(point)
    #print(area)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return area

def circle_area(img):
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('img',800,600)
    # blue
    lower_blue = np.array ([100, 90, 50])
    upper_blue = np.array ([130,255, 255])
    mask_b=cv2.inRange(img_hsv,lower_blue,upper_blue)
    img=cv2.bitwise_and(img,img,mask=mask_b)
    cv2.imshow('img',mask_b)
    contour = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    c_contour_sort = sorted(contour, key=cv2.contourArea, reverse=True)[0]
    # print(c_contour_sort)
    # cv2.drawContours(img, c_contour_sort, -1, (0, 0, 255), 3)
    # cv2.imshow('img_c',img_c)
    points = np.array(c_contour_sort, np.float32)
    points = points.reshape((-1, 1, 2))
    area = cv2.contourArea(points)
    # print(area)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return area

folder_path = 'D:/learn/train_2024/train1/testpic'
rect_out = []
cir_out  = []
w=156
h=76
s=w*h
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        input_img = cv2.imread(file_path)
        m_area = int(money_area(input_img))
        # print("money",m_area)
        s_area = int(square_area(input_img))
        # print("square",s_area)
        c_area = int(circle_area(input_img))
        true_square=s_area/m_area*s
        print(true_square)
        l=math.sqrt(true_square)
        true_circle=c_area*s/m_area
        d=math.sqrt(true_circle/math.pi)*2
        rect_out.append(str(l))
        cir_out.append(str(d))
print("Square")
print(rect_out)
print("Circle")
print(cir_out)
with open("cir.txt", "w") as file1:
    for each in cir_out:
        file1.write(each+'\n')

with open("rect.txt", "w") as file2:
    for rect in rect_out:
        #print(rect)
        file2.write(rect+'\n')