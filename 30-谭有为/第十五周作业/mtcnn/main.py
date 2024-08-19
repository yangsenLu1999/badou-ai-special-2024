import cv2
import mtcnn_model

img=cv2.imread('F:/PNG/face_detect/face2.jpg')


model=mtcnn_model.MTCNN()
thresholds=[0.5,0.6,0.8]
rectangles=model.detect_face(img,thresholds)
print('count-face',len(rectangles))
img1=img.copy()

for rectangle in rectangles:
    if rectangle is not None:
        w=int(rectangle[2])-int(rectangle[0])
        h=int(rectangle[3])-int(rectangle[1])
        paddingW=0.01*w
        paddingH=0.02*h  # 根据宽度和高度计算水平和垂直方向的填充量（padding）。具体来说，这段代码用于在裁剪人脸区域时增加一些额外的像素，以确保裁剪的区域包含更多的人脸信息
        crop_img=img[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH),
                 int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]  # 裁剪人脸区域
        if crop_img is None:  # 确保裁剪区域不为空
            continue
        if crop_img.shape[0]<0 or crop_img.shape[1]<0:  # 确保裁剪区域不为空
            continue
        cv2.rectangle(img1,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)  # 绘制矩形框

        for i in range(5,15,2):
            cv2.circle(img1,(int(rectangle[i]),int(rectangle[i+1])),(2),(0,255,0),1)

#cv2.imwrite('F:/PNG/face_detect/out.jpg',img1)
cv2.imshow('face_detect',img1)
cv2.waitKey(0)

