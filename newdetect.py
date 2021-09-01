import argparse # python的命令行解析的模块，内置于python，不需要安装
import os
import shutil
import time
from pathlib import Path
import pyrealsense2 as rs

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    # 获取out(输出文件夹)、source(输入源)、weights(权重)、imgsz等参数
    out, source, weights, view_img, save_txt, imgsz = \
        opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    # 获取设备
    device = select_device(opt.device)
    # 删除之前的输出文件夹
    if os.path.exists(out): # output dir
        shutil.rmtree(out)  # delete dir
    os.makedirs(out)  # make new dir
    # 如果设备为gpu，使用Float16
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # 加载Float32模型，确保用户设定的输入图片分辨率能整除32(如不能则调整为能整除并返回)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    # 设置Float16
    if half:
        model.half()  # to FP16

    # Set Dataloader
    # 通过不同的输入源来设置不同的数据加载方式
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        # 加载图片或视频
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    # 获取类别名字字符串列表
    names = model.module.names if hasattr(model, 'module') else model.names
    # 设置画框的颜色(RGB值（列表）的列表)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    # 进行一次前向推理,测试程序是否正常
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    """
    path 图片/视频路径
    img 进行resize+pad之后的图片, 如(3,384,640) 格式(c,h,w)
    img0s 原size图片，如(720,1280,3)
    vid_cap 当读取图片时为None，读取视频时为视频源
    """
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device) # Tensor
        # 图片也设置为Float16
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 ~ 255 to 0.0 ~ 1.0
        # 没有batch_size时，在最前面添加一个轴
        if img.ndimension() == 3:
            img = img.unsqueeze(0) # shape 如 (1,3,384,640)

        # Inference
        t1 = time_synchronized()
        """
        前向传播 返回pred的shape是(1, num_boxes, 5+num_class)
        h,w为传入网络图片的高和宽。注意dataset在检测时使用了矩形推理，所以这里h不一定等于w
        num_boxes = (h/32 * w/32 + h/16 * w/16 + h/8 * w/8)*3
        例如：图片大小720,1280-->15120个boxes = (20*12 + 40*24 + 80*48=5040)*3
        pred[..., 0:4]为预测框坐标; 预测框坐标为xywh(中心点+宽高)格式
        pred[..., 4]为objectness置信度得分
        pred[..., 5:-1]为分类概率结果
        """
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        # 进行NMS
        """
        pred: 前向传播的输出
        conf_thres: 置信度阈值
        iou_thres: iou阈值
        classes: 是否只保留特定的类别
        agnostic_nms: 进行nms是否也去除不同类别之间的框
        经过nms之后，预测框格式：xywh-->xyxy(左上角右下角)
        pred是一个列表list[torch.tensor]，长度为NMS后的目标框的个数
        每一个torch.tensor的shape为(num_boxes, 6),内容为box(4个值)+conf+cls
        """
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        # 对每一张图片作处理
        for i, det in enumerate(pred):  # detections per image
            # 如果输入源是webcam，则batch_size不为1，取出dataset中的一张图片
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            # 设置保存图片或视频的路径
            # p是原图片路径（含文件名）
            save_path = str(Path(out) / Path(p).name)
            # 设置保存框坐标txt文件的路径
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            # 设置打印信息(图片宽高)， s 如‘384*640’
            s += '%gx%g ' % img.shape[2:]  # print string
            # gn 如[810,1080,810,1080]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                # 此时坐标格式为xyxy
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 打印检测到的类别数量
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                # 保存预测结果
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽高)格式，并除上w，h做归一化，转化为列表再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line) + '\n') % line)  # label format
                    # 在原图上画框
                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            # 打印前向传播+nms时间
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            # 如果设置展示，则画出图片/视频
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            # 设置保存图片/视频
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))

    # 打印总时间
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    # 建立参数解析对象parser
    parser = argparse.ArgumentParser()
    """
    weights: 训练的权重
    source: 测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
    img-size: 网络输入图片大小
    conf-thres: 置信度阈值
    iou-thres: 做nms的iou阈值
    device: 设置设备
    view-img: 是否展示预测之后的图片/视频，默认False
    save-txt: 是否将预测的框坐标以txt文件形式保存，默认False
    save-conf: 是否将预测的框置信度以txt文件形式保存，默认False
    save-dir: 网络预测之后的图片/视频的保存路径
    classes: 设置只保留某一部分类别，形如0或者0 2 3
    agnostic-nms: 进行nms是否也去除不同类别之间的框，默认False
    augment: 推理的时候进行多尺度，翻转等操作(TTA)推理
    update: 如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    """
    # 添加属性：给xx实例增加一个aa属性，如 xx.add_argument("aa")
    # nargs - 应该读取的命令行参数个数。*号，表示0或多个参数；+号表示1或多个参数。
    # action - 命令行遇到参数时的动作。action=‘store_true’，只要运行时该变量有传参就将该变量设为True。
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    # 采用parser对象的parse_args函数获取解析的参数
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad(): # 一个上下文管理器，被该语句wrap起来的部分将不会track梯度
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                # 去除pt文件中的优化器等信息
                strip_optimizer(opt.weights)
        else:
            detect()
