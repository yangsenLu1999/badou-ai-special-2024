import numpy as np


def  Scale(img):   #缩放比例函数
    copy_img = img.copy()
    pr_scale = 1.0
    h, w, c= copy_img.shape
    if min(w, h) > 500:
        pr_scale = 500.0 / min(h, w)
        w = int(w * pr_scale)
        h = int(h * pr_scale)

    elif max(w, h) < 500:
        pr_scale = 500.0 / max(h, w)
        w = int(w * pr_scale)
        h = int(h * pr_scale)

    scales=[]
    factor=0.709  #缩放因子，初始值0.709
    factor_count=0
    minl=min(h,w)  #图像最小边长
    while minl >= 12:
        scales.append(pr_scale*pow(factor, factor_count))  #pow--->幂运算
        minl *= factor
        factor_count += 1  #通过逐步缩小图像的最小边长（minl），并计算相应的比例因子，直到最小边长小于12为止
    return scales


def NMS(rectangles,threshold):
    if len(rectangles)==0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort())
    pick = []
    while len(I)>0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) #I[-1] 表示最大得分的框，I[0:-1]表示其他框
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o<=threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle

def rect2square(rectangles):  #长方形调整为正方形
    w=rectangles[:,2]-rectangles[:,0]
    h=rectangles[:,3]-rectangles[:,1]
    l=np.maximum(w,h).T  #每个长方形调整为正方形后的边长，取宽度和高度的最大值。
    rectangles[:,0]=rectangles[:,0]+w*0.5-l*0.5
    rectangles[:,1]=rectangles[:,1]+h*0.5-l*0.5
    rectangles[:,2:4]=rectangles[:,0:2]+np.repeat([l],2,axis=0).T  #左上角坐标+边长，得到右下角坐标
    return rectangles


def detect_face_pnet(cls_prob, roi, out_side, scale,width,height,threshold):   #对Pnet的输出处理
    cls_prob = np.swapaxes(cls_prob, 0, 1) # 交换维度
    roi = np.swapaxes(roi, 0, 2)

    stride = 0
    if out_side != 1:
        stride = float(2*out_side-1)/(out_side-1) #计算步长，约等于2
    (x, y) = np.where(cls_prob >= threshold)  # 得到大于阈值的坐标

    boudingbox=np.array([x,y]).T
    #将特征图上的边界框坐标映射回原图，并计算边界框的左上角和右下角坐标。
    bb1=np.fix((stride * (boudingbox) + 0) * scale)  # 得到框的左上角坐标   np.fix--->取整
    bb2=np.fix((stride * (boudingbox) + 11) * scale)  # 得到框的右下角坐标
    boudingbox=np.concatenate((bb1,bb2),axis=1)  # 得到框的左上角和右下角坐标

    dx1=roi[0][x,y]
    dx2=roi[1][x,y]
    dx3=roi[2][x,y]
    dx4=roi[3][x,y]
    score=np.array([cls_prob[x,y]]).T   # 得到框的得分
    offset=np.array([dx1,dx2,dx3,dx4]).T   # 得到框的偏移量

    boudingbox=boudingbox+offset*12.0*scale   # 得到框的左上角和右下角坐标，12表示框的大小，scale表示缩放比例

    rectangles=np.concatenate((boudingbox,score),axis=1)   # 得到框的左上角和右下角坐标和得分
    rectangles=rect2square(rectangles)   # 调整框的长宽比,使其为正方形
    pick=[]
    for i in range(len(rectangles)):
        #print('ss',rectangles[i])
        x1=int(max(0,rectangles[i][0]))
        y1=int(max(0,rectangles[i][1]))  # 防止框的坐标为负数
        x2=int(min(width,rectangles[i][2]))
        y2=int(min(height,rectangles[i][3]))  # 防止框超出图像边界
        score=rectangles[i][4] # 得到框的得分
       # print('5canshu',x1,y1,x2,y2,score)
        if x2-x1>0 or y2-y1>0:  # 防止框的宽度或高度为0
            pick.append([x1,y1,x2,y2,score])
    return NMS(pick,0.3)   # 非极大值抑制，阈值为0.3


def detect_face_rnet(cls_prob, roi, rectangles, width,height,threshold):   #对Rnet的输出处理
    prob=cls_prob[:,1]  #   得到Rnet的概率值
    pick=np.where(prob>=threshold)  # 得到大于阈值的坐标
    rectangles=np.array(rectangles)

    x1=rectangles[pick,0]
    y1=rectangles[pick,1]
    x2=rectangles[pick,2]
    y2=rectangles[pick,3]  # 得到框的左上角和右下角坐标
    score=np.array([prob[pick]]).T  # 得到框的得分

    dx1=roi[pick,0]
    dx2=roi[pick,1]
    dx3=roi[pick,2]
    dx4=roi[pick,3]  # 得到框的偏移量

    w=x2-x1
    h=y2-y1

    x1=np.array([(x1+dx1*w)[0]]).T
    y1=np.array([(y1+dx2*h)[0]]).T
    x2=np.array([(x2+dx3*w)[0]]).T
    y2=np.array([(y2+dx4*h)[0]]).T   #   得到框在原图上的的左上角和右下角坐标

    rectangles=np.concatenate((x1,y1,x2,y2,score),axis=1)  # 得到框的左上角和右下角坐标和得分
    rectangles=rect2square(rectangles)   # 调整框的长宽比,使其为正方形
    pick=[]
    for i in range(len(rectangles)):
        x1=int(max(0,rectangles[i][0]))
        y1=int(max(0,rectangles[i][1]))  # 防止框的坐标为负数
        x2=int(min(width,rectangles[i][2]))
        y2=int(min(height,rectangles[i][3]))  # 防止框超出图像边界
        score=rectangles[i][4] # 得到框的得分
        if x2-x1>0 or y2-y1>0:  # 防止框的宽度或高度为0
            pick.append([x1,y1,x2,y2,score])
    print('rnet-result:',NMS(pick,0.3))
    return NMS(pick,0.3)   # 非极大值抑制，阈值为0.3


def detect_face_onet(cls_prob, roi,pts, rectangles, width,height,threshold):   #对Onet的输出处理,pts是关键点坐标的数组

    prob=cls_prob[:,1]  #   得到Onet的概率值
    pick=np.where(prob>=threshold)  # 得到大于阈值的坐标
    rectangles=np.array(rectangles)

    x1=rectangles[pick,0]
    y1=rectangles[pick,1]
    x2=rectangles[pick,2]
    y2=rectangles[pick,3]  # 得到框的左上角和右下角坐标

    score=np.array([prob[pick]]).T  # 得到框的得分
    #print(x1,y1,x2,y2,score)

    dx1=roi[pick,0]
    dx2=roi[pick,1]
    dx3=roi[pick,2]
    dx4=roi[pick,3]  # 得到框的偏移量

    w=x2-x1
    h=y2-y1

    pts0=np.array([(w*pts[pick,0]+x1)[0]]).T  # 得到关键点坐标并转换到原图
    pts1=np.array([(h*pts[pick,5]+y1)[0]]).T
    pts2=np.array([(w*pts[pick,1]+x1)[0]]).T
    pts3=np.array([(h*pts[pick,6]+y1)[0]]).T
    pts4=np.array([(w*pts[pick,2]+x1)[0]]).T
    pts5=np.array([(h*pts[pick,7]+y1)[0]]).T
    pts6=np.array([(w*pts[pick,3]+x1)[0]]).T
    pts7=np.array([(h*pts[pick,8]+y1)[0]]).T
    pts8=np.array([(w*pts[pick,4]+x1)[0]]).T
    pts9=np.array([(h*pts[pick,9]+y1)[0]]).T

    x1=np.array([(x1+dx1*w)[0]]).T
    y1=np.array([(y1+dx2*h)[0]]).T
    x2=np.array([(x2+dx3*w)[0]]).T
    y2=np.array([(y2+dx4*h)[0]]).T   #   得到框在原图上的的左上角和右下角坐标
    rectangles=np.concatenate((x1,y1,x2,y2,score,pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9),axis=1)  # 得到框的左上角和右下角坐标和得分
    print('onet-rectangles:',rectangles)
    pick=[]
    for i in range(len(rectangles)):
        x1=int(max(0,rectangles[i][0]))
        y1=int(max(0,rectangles[i][1]))  # 防止框的坐标为负数
        x2=int(min(width,rectangles[i][2]))
        y2=int(min(height,rectangles[i][3]))  # 防止框超出图像边界
        if x2-x1>0 or y2-y1>0:  # 防止框的宽度或高度为0
            pick.append([x1,y1,x2,y2,rectangles[i][4],
                         rectangles[i][5],rectangles[i][6],rectangles[i][7],
                         rectangles[i][8],rectangles[i][9],rectangles[i][10],
                         rectangles[i][11],rectangles[i][12],rectangles[i][13],
                         rectangles[i][14]])

    return NMS(pick,0.3)   # 非极大值抑制，阈值为0.3
























