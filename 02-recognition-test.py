#!/usr/bin/env python
# coding: utf-8

# In[1]:


from detect_yolo import detect_yolo_from_image


# In[2]:


#import numpy as np


# In[3]:


from coco_utils import convert_to_coco_dict
from coco_utils import convert_to_coco_json
from coco_utils import convert_bbox_to_metric, convert_metric_to_box
from coco_utils import check_result_coco
from coco_utils import metric_map
from coco_utils import load_coco_file


# In[4]:


from mean_average_precision import MetricBuilder


# In[5]:


resultFile='dataset/submission_example.json'


# In[6]:


result_coco = load_coco_file(resultFile)


# In[7]:


result_coco = check_result_coco(result_coco)


# In[8]:


folder = 'dataset/test/images/'


# In[9]:


result = detect_yolo_from_image(folder)


# In[10]:


#Запишем результаты по изображениям
for i in result_coco.imgs:
    d_img = result_coco.imgs[i]
    result_bbox = []
    result_confidence = []
    for j in range(len(result)):
        bbox, confidence, filename = result[j]
        if d_img['file_name'] == filename:
            print(filename, bbox)
            result_bbox.append(bbox)
            result_confidence.append(confidence)
    result_coco.imgs[i]['predict_bbox'] = result_bbox
    result_coco.imgs[i]['predict_confidence'] = result_confidence


# In[11]:


one_sample  = result_coco.dataset['annotations'][1]
do_ann = result_coco.dataset['annotations']


# In[12]:


result_coco.dataset['annotations'] = []


# In[13]:


#Записываем результаты
ann_id = 0
for i in result_coco.imgs:
    d_img = result_coco.imgs[i]
    result_bbox = []
    result_confidence = []
    result_bbox = d_img['predict_bbox']
    result_confidence = d_img['predict_confidence']
    
    not_ann = True
    for j in range(len(result_confidence)):
        bbox = result_bbox[j]
        confidence = result_confidence[j]
        
        ann_id += 1
    
        one_sample['id'] = ann_id
        one_sample['image_id'] = d_img['id']
        
        one_sample['bbox'] = bbox
        (x, y, width, height) = bbox
        one_sample['area'] = width*height
        
        result_coco.dataset['annotations'].append(dict(one_sample))
        
        not_ann = False
        
    if not_ann:
        ann_id += 1
        one_sample['id'] = ann_id
        one_sample['image_id'] = d_img['id']
        one_sample['bbox'] = (0,0,0,0)
        one_sample['area'] = 0
        
        result_coco.dataset['annotations'].append(dict(one_sample))


# In[15]:


for i in result_coco.imgs:
    result_coco.imgs[i].pop('predict_bbox')
    result_coco.imgs[i].pop('predict_confidence')


# In[21]:


filename =  'result_file.json'


# In[22]:


d = convert_to_coco_json(result_coco, filename, allow_cached=True)


# In[23]:


test_coco = load_coco_file(filename)


# In[ ]:




