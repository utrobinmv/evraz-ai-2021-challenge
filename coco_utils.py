import json
import datetime
import numpy as np

from pycocotools.coco import COCO

from mean_average_precision import MetricBuilder

def load_coco_file(resultFile):
    '''
    Чтение coco файла
    '''
    result_coco = COCO(resultFile)
    return result_coco

def convert_bbox_to_metric(bbox):
    '''
    Ковертирует bbox в mbox
    '''
    (x, y, width, height) = bbox
    mbox = (x,y,x+width, y+height)
    return mbox

def convert_metric_to_box(mbox):
    '''
    Ковертирует mbox в bbox
    '''
    (x1, y1, x2, y2) = mbox
    width = x2 - x1
    height = y2 - y1
    bbox = (int(x1),int(y1),int(width), int(height))
    return bbox

def metric_map(list_frames):
    '''
    Подсчет метрики mAP
    '''
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", num_classes=1)
    num_frames = len(list_frames)
    for i in range(num_frames):
        preds, gt = list_frames[i]
        #print(preds[i], gt[i])
        metric_fn.add(preds, gt)
    mAP = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05))
    return mAP 

def check_result_coco(coco):
    '''
    Исправляет ошибки в входной coco файл
    '''
    for i in coco.anns:
        ann = coco.anns[i]
        bbox = ann['bbox']

        (x, y, width, height) = bbox

        alarm = False
        if not isinstance(x, int): 
            alarm = True
            x = 0
        if not isinstance(y, int): 
            alarm = True
            y = 0
        if not isinstance(width, int): 
            alarm = True
            width = 1
        if not isinstance(height, int): 
            alarm = True
            height = 1
        if x<0:
            alarm = True
            x = 0
        if y<0:
            alarm = True
            y = 0
        if width<0:
            alarm = True
            width = 1
        if height<0:
            alarm = True
            height = 1
        if x > 1920:
            x = 0
            alarm = True
        if y > 1080:
            y = 0
            alarm = True
        if x+width > 1920:
            width = 1
            alarm = True
        if y+height > 1080:
            height = 1
            alarm = True

        if alarm:
            print('Ошибка:',x, y, width, height)

        ann['bbox'] = (x, y, width, height)
    
    return coco

def convert_to_coco_dict(coco):
    '''
    Генерирует контекст файла coco
    '''
    coco_images = []
    coco_annotations = []
    
    dataset_images = coco.dataset['images']
    for coco_image in dataset_images:
        coco_images.append(coco_image)
    dataset_anns = coco.dataset['annotations']
    for annotation in dataset_anns:
        #annotation['area'] = 0
        coco_annotations.append(annotation)

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "",
    }
    
    coco_dict = {"info": info, "images": coco_images, "categories": coco.dataset['categories'], "licenses": None}
    if len(coco_annotations) > 0:
        coco_dict["annotations"] = coco_annotations
    return coco_dict


def convert_to_coco_json(dataset_name, output_file, allow_cached=True):
    '''
    Записывает файл coco в json
    '''
    coco_dict = convert_to_coco_dict(dataset_name)
    with open(output_file, 'w') as f:
        json.dump(coco_dict, f,  ensure_ascii=False)
    return coco_dict
