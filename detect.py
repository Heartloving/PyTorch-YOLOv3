from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

# 测试图片和类名的文件夹路径信息
image_folder = "/media/xddz/本地磁盘/Learning/CODE_DATA/DATA/yolov3_data/samples"
class_path = "/media/xddz/本地磁盘/Learning/CODE_DATA/DATA/yolov3_data/coco.names"

if __name__ == "__main__":
    # 创建一个ArgumentParser对象，格式: 参数名, 目标参数(dest是字典的key),帮助信息,默认值,类型
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default=image_folder,help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str,default= class_path ,help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    # 输出结果是一个namespace类型的变量，即argparse.Namespace, 可以像easydict一样使用,就像一个字典，key来索引变量的值
    # Namespace(bs=1, cfgfile='cfg/yolov3.cfg', confidence=0.5,det='det', images='imgs', nms_thresh=0.4, reso='416', weightsfile='yolov3.weights')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#选择GPU 或是cpu

    os.makedirs("output", exist_ok=True)#创建output文件夹


    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)#传入网络参数初始化darknet模型，并传入计算设备

    if opt.weights_path.endswith(".weights"):# 将权重文件载入，并复制给对应的网络结构model中
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode
    # 变成测试模式，这主要是对dropout和batch normalization的操作在训练和测试的时候是不一样的

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )#加载待检测图像


    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths

    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time() #计时节点
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # enumerate返回im_batches列表中每个batch在0维连接成一个元素的tensor和这个tensor在im_batches中的序号。

        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        # Variable(batch)将图片生成一个可导tensor

        # Get detections
        with torch.no_grad():#检测过程不需要梯度回传
            detections = model(input_imgs)#传入图片进行检测
            # prediction是一个batch所有图片通过yolov3模型得到的预测值，维度为1x10647x85，三个scale的图片
            # 每个scale的特征图大小为13x13,26x26,52x52,一个元素看作一个格子，每个格子有3个anchor，将一个anchor保存为一行，
            # 所以prediction一共有(13x13+26x26+52x52)x3=10647行，一个anchor预测(x,y,w,h,s,s_cls1,s_cls2...s_cls_80)，
            # 一共有85个元素。所以prediction的维度为Bx10647x85，加为这里batch_size为1，所以prediction的维度为1x10647x85

            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)#非极大抑制
            #输出n行7列，表示检测出n个目标，7个参数分别是四个位置坐标，两个置信度，一个类别。


        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)#计算每张图片的检测消耗时间
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)


    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]#构建颜色列表


    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:

            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2]) #将目标框对应到原图
            unique_labels = detections[:, -1].cpu().unique()#目标框类别列进行去重操作并重新组成列表

            n_cls_preds = len(unique_labels)#统计类别数量
            bbox_colors = random.sample(colors, n_cls_preds)#随机获得类别数目的颜色
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1#宽
                box_h = y2 - y1#高

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]#选择类别所对应的颜色
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")#画矩形
                # Add the bbox to the plot
                ax.add_patch(bbox)#添加矩形框
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )#添加类别文字说明

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())#去掉白边
        plt.gca().yaxis.set_major_locator(NullLocator())#去掉白边
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0) #保存图片
        #f-string是python3.6引入的，用大括号{}来标明被替换的字段
        plt.close()
