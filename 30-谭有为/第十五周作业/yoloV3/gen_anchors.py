import numpy as np
import matplotlib.pyplot as plt
from pycocotools import COCO

#将标注框的x1,y1,x2,y2转换为x，y,w,h格式 并归一化到0-1之间，x,y表示中心点坐标
def convert_coco_bbox(size,box):  #size:原始图像大小 （512,512）   box：标注box的信息（x1,y1,w,h）或者（x1,y1,x2,y2）
    dw=1./size[0]
    dh=1./size[1]
    x=(box[0]+box[2])/2
    y=(box[1]+box[3])/2
    w=box[2]
    h=box[3]
    x=x*dw
    w=w*dw
    y=y*dh
    h=h*dh
    return x,y,w,h

#计算每个box和聚类中心的距离值
def box_iou(boxes,clusters):   #boxes:所有框的坐标 （[x1,y1,w1,h1],[x2,y2,w2,h2],...）  cluster：聚类中心 ([x1,y1],[x2,y2]....)
    box_num=boxes.shape[0]
    cluster_num=clusters.shape[0]
    box_area=boxes[:,0]*boxes[:,1]
    box_area=box_area.repeat(cluster_num)                           #repeat函数---生成指定的维度重复张量
    box_area=np.reshape(box_area,[box_num,cluster_num])

    cluster_area=clusters[:,0]*clusters[:,1]
    cluster_area=np.tile(cluster_area,[1,cluster_num])   #np.tile 对数组进行重复操作，参数（A，reps）,reps为一个列表，表示对各axis的重复次数
    cluster_area=np.reshape(cluster_area,[box_num,cluster_num])

    # #这里计算两个矩形的iou，默认所有矩形的左上角坐标都是在原点，然后计算iou，因此只需取长宽最小值相乘就是重叠区域的面积
    boxes_w=np.reshape(boxes[:,0].repeat(cluster_num),[box_num,cluster_num])
    cluster_w=np.reshape(clusters[:,0].repeat(box_num),[box_num,cluster_num])
    min_w=np.minimum(cluster_w,boxes_w)

    boxes_h=np.reshape(boxes[:,1].repeat(cluster_num),[box_num,cluster_num])
    cluster_h=np.reshape(clusters[:,1].repeat(box_num),[box_num,cluster_num])
    min_h=np.minimum(cluster_h,boxes_h)
#np.multiply，元素乘法，即数组A（a,b） B(c,d),则 np.multiply=（（a*c），（b*d））
    iou=np.multiply(min_h,min_w)/(box_area+cluster_area-(np.multiply(min_h,min_w)))  #iou=并集/交集

    return iou

#计算所有box和聚类中心的最大iou均值作为准确率
def avg_iou(boxes,clusters):
    iou=box_iou(boxes,clusters)
    max_iou=np.max(iou,axis=1)
    avg_iou=np.mean(max_iou)
    return avg_iou

#根据所有box的长宽进行Kmeans聚类
def Kmeans(boxes,clusters_num,iteration_cutoff=25,function=np.median):
    #np.median---返回数组的中位数  ,iteration_cutoff表示准确率不在降低后 多少轮开始停止迭代
    box_num=boxes.shape[0]
    best_avg_iou=0
    best_avg_iou_interation=0
    best_cluster=[]
    anchors=[]
    np.random.seed()
    #随机选择几个box作为聚类中心
    clusters=boxes[np.random.choice(box_num,clusters_num,replace=False)]  #np.random.choice---随机抽样函数，参数replace 表示是否重复抽样
    count=0
    while True:
        distance=1.-box_iou(boxes,clusters)
        # 获取每个box距离哪个聚类中心最近
        boxes_iou=np.min(distance,axis=1)
        current_box_cluster=np.argmin(distance,axis=1)
        average_iou=np.mean(1.-boxes_iou)
        if average_iou>best_avg_iou:
            best_avg_iou=average_iou
            best_cluster=clusters
            best_avg_iou_interation=count
         #更新聚类中心
        for c in range(clusters_num):
            clusters[c]=function(boxes[current_box_cluster==clusters],axis=0)
        if count>=best_avg_iou_interation+iteration_cutoff:
            break
        print("Sum of all distances (cost) = {}".format(np.sum(boxes_iou)))
        print("iter: {} Accuracy: {:.2f}%".format(count, avg_iou(boxes, clusters) * 100))
        count+=1

    for cluster in best_cluster:
        anchors.append([round(clusters[0]*416),round(clusters[1]*416)])
    return anchors,best_avg_iou


#读取coco数据集
def load_cocoDataset(ann_file):
    data=[]
    coco=COCO(ann_file)
    cats=coco.loadCats(coco.getCatIds())
    coco.loadImgs()
    base_classes={cat['id']:cat['name'] for cat in cats}
    imgId_catIds=[coco.getImgIds(catIds=cat_ids) for  cat_ids in base_classes.keys()]
    image_ids=[img_id for img_cat_id in imgId_catIds for img_id in img_cat_id ]
    for image_id in image_ids:
        annIds=coco.getAnnIds(imgIds=image_id)
        anns=coco.loadAnns(annIds)
        img=coco.loadImgs(image_id)[0]
        image_w=img['width']
        image_h=img['height']

    for ann in anns:
        box=ann['bbox']
        bb=convert_coco_bbox((image_w,image_h),box)
        data.append(bb[2:])
    return np.array(data)

#主处理函数
def process(dataFile,cluster_num,iteration_cutoff=25,function=np.median):
    last_best_iou=0
    last_anchors=[]
    boxes=load_cocoDataset(dataFile)
    box_w=boxes[:1000,0]
    box_h=boxes[:1000,1]
    plt.scatter(box_w,box_h,c='r')
    anchors=Kmeans(boxes,cluster_num,iteration_cutoff=iteration_cutoff,function=function)
    plt.scatter(anchors[:,0],anchors[:,1],c='g')
    plt.show()
    for i in range(100):
        anchors,best_iou=Kmeans(boxes,cluster_num,iteration_cutoff,function)
        if best_iou>last_best_iou:
            last_anchors=anchors
            last_best_iou=best_iou
            print("anchors: {}, avg iou: {}".format(last_anchors, last_best_iou))
    print("final anchors: {}, avg iou: {}".format(last_anchors, last_best_iou))


if __name__=='__main__':
    process('',9)























