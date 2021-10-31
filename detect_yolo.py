# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import random
import shutil
import time
import gc

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import yaml

from yolov4_pytorch.data import LoadImages
from yolov4_pytorch.data import LoadStreams
from yolov4_pytorch.data import check_image_size
from yolov4_pytorch.model import YOLO
from yolov4_pytorch.model import apply_classifier
from yolov4_pytorch.model import load_classifier
from yolov4_pytorch.utils import non_max_suppression
from yolov4_pytorch.utils import plot_one_box
from yolov4_pytorch.utils import scale_coords
from yolov4_pytorch.utils import select_device
from yolov4_pytorch.utils import time_synchronized
from yolov4_pytorch.utils import xyxy2xywh

def detect_yolo_from_image(source):
    '''
    Это переработанный код из репозитория
    https://github.com/Lornatang/YOLOv4-PyTorch
    Адаптированный под нашу задачу
    
    Функция выполняет детекцию изображний из каталога.
    Может работать и с одним изображением
    
    Преимущество фукнции что она может работать как с картинками так и с видеопотоком
    '''
    
    
    config_file = 'configs/yolov4m.yaml'
    data = "data/voc2007.yaml"
    output = 'outputs'
    weights = 'weights/yolov4-45544c52.pth'
    weights = 'weights/model_best-.pth'
    weights = 'weights/model_best-2021-10-30_15_36.pth'
    weights = 'weights/model_best-2021-10-30_19-24.pth'
    weights = 'weights/model_best.pth'
    view_image = False
    save_txt = False
    save_image = False
    confidence_thresholds = 0.40 #0.09 #0.4
    iou_thresholds = 0.5 #0.5 ?0.4
    classes = None
    agnostic_nms = None
    device = '0'
    augment = False
    image_size = 1024
    #image_size = 1080
    #image_size = 640
    save_out = False
    save_result = True
    
    result = []
    
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    number_classes, names = int(data_dict["number_classes"]), data_dict["names"]
    
    camera = False
    if source == "0" or source.startswith("rtsp") or source.startswith("http") or source.endswith(".txt"):
        camera = True

    # Initialize
    device = select_device(device)
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder
    
    half = False
    
    #half = device.type != "cpu"  # half precision only supported on CUDA

    # Create model
    model = YOLO(config_file=config_file, number_classes=number_classes).to(device)
    image_size = check_image_size(image_size, stride=32)

    w = torch.load(weights)

    # Load model
    model.load_state_dict(w["state_dict"])
    model.float()
    model.fuse()
    model.eval()
    if half:
        model.half()  # to FP16

    model_classifier = None

    # Set Dataloader
    video_path, video_writer = None, None
    if camera:
        view_image = True
        cudnn.benchmark = True
        dataset = LoadStreams(source, image_size=image_size)
    else:
        if save_out:
            save_image = True
        dataset = LoadImages(source, image_size=image_size)

    # Get names and colors
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    start_time = time.time()

    image = torch.zeros((1, 3, image_size, image_size), device=device)  # init image
    _ = model(image.half() if half else image) if device.type != "cpu" else None  # run once
    for filename, image, raw_images, video_capture in dataset:
        image = torch.from_numpy(image).to(device)
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        # Inference
        inference_time = time_synchronized()
        pred = model(image, augment=augment)[0]
        
        # Apply NMS
        prediction = non_max_suppression(prediction=pred.detach().cpu(),
                                         confidence_thresholds=confidence_thresholds,
                                         iou_thresholds=iou_thresholds,
                                         classes=classes,
                                         agnostic=agnostic_nms)
        nms_time = time_synchronized()
        
        # Process detections
        for i, detect in enumerate(prediction):  # detections per image
            if camera:  # batch_size >= 1
                p, context, raw_image = filename[i], f"{i:g}: ", raw_images[i].copy()
            else:
                p, context, raw_image = filename, "", raw_images

            save_path = os.path.join(output, p.split("/")[-1])
            txt_filename = f"_{dataset.frame if dataset.mode == 'video' else ''}"
            txt_path = os.path.join(output, p.split("/")[-1][-4:] + txt_filename)

            context += f"{image.shape[2]}*{image.shape[3]} "  # get image size

            gn = torch.tensor(raw_image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if detect is not None and len(detect):
                # Rescale boxes from img_size to im0 size
                detect[:, :4] = scale_coords(image.shape[2:], detect[:, :4], raw_image.shape).round()

                # Print results
                for category in detect[:, -1].unique():
                    # detections per class
                    if int(category) != 14: #Нас интересуют только люди
                        continue
                    
                    number = (detect[:, -1] == category).sum()
                    if number > 1:
                        context += f"{number} {names[int(category)]}s, "
                    else:
                        context += f"{number} {names[int(category)]}, "

                # Write results
                for *xyxy, confidence, classes_id in detect:
                    if int(classes_id) != 14: #Нас интересуют только люди
                        continue
                    
                    if save_result:
                        mbox = list(torch.tensor(xyxy).view(1, 4).cpu().numpy())[0]
                        (x1, y1, x2, y2) = mbox
                        width = x2 - x1
                        height = y2 - y1
                        bbox = (int(x1),int(y1),int(width),int(height))                        
                        
                        result.append((bbox, confidence.detach().cpu().numpy(), p.split("/")[-1]))
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + ".txt", "a") as f:
                            f.write(f"{classes_id} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n")

                    if save_image or view_image:  # Add bbox to image
                        label = f"{names[int(classes_id)]} {int(confidence * 100)}%"
                        plot_one_box(xyxy=xyxy,
                                     image=raw_image,
                                     color=colors[int(classes_id)],
                                     label=label,
                                     line_thickness=3)

            # Stream results
            if view_image:
                cv2.imshow("camera", raw_image)
                if cv2.waitKey(1) == ord("q"):  # q to quit
                    raise StopIteration

            # Print time (inference + NMS)
            print(f"{context}Done. {nms_time - inference_time:.3f}s")

            # Save results (image with detections)
            if save_image:
                if dataset.mode == "images":
                    cv2.imwrite(save_path, raw_image)
                else:
                    if video_path != save_path:  # new video
                        video_path = save_path
                        if isinstance(video_writer, cv2.VideoWriter):
                            video_writer.release()  # release previous video writer

                        fourcc = "mp4v"  # output video codec
                        fps = video_capture.get(cv2.CAP_PROP_FPS)
                        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    video_writer.write(raw_image)

    print(f"Done. ({time.time() - start_time:.3f}s)")

    return result
