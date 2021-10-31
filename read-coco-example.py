#!/usr/bin/env python
# coding: utf-8

# In[1]:


from detect_yolo import detect_yolo_from_image


# In[2]:


from pycocotools.coco import COCO


# In[3]:


#from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt


# In[4]:


#!pip install --upgrade numpy


# In[5]:


#!pip install opencv-python==4.3.0.38
#!pip install torch==1.10.0
#!pip install -r my_req.txt
#!pip uninstall -y pycocotools
#!pip install pycocotools
#!pip uninstall -y numpy
#!pip install numpy


# In[6]:


#!pip install torch==1.9.1


# In[7]:


#import torch


# In[8]:


#torch.__version__


# In[9]:


#import torch.backends.cudnn as cudnn


# In[10]:


#torch.backends.cudnn.version()


# In[11]:


#from detect_yolo import detect_yolo_from_image


# In[12]:


annFile='dataset/train/annotations/COCO_json/coco_annotations_train.json'


# In[13]:


coco=COCO(annFile)


# In[14]:


list(coco.__dict__)


# In[15]:


len(coco.imgs)


# In[16]:


len(coco.imgToAnns)


# In[17]:


coco.cats


# In[18]:


len(coco.anns)


# In[19]:


list(coco.dataset)


# In[20]:


coco.dataset['info']


# In[21]:


coco.dataset['licenses']


# In[22]:


catIds = coco.getCatIds(catNms=['person']);


# In[23]:


catIds


# In[24]:


imgIds = coco.getImgIds(catIds=catIds );


# In[25]:


len(imgIds)


# In[26]:


img = coco.loadImgs(imgIds[1])[0]


# In[27]:


img


# In[28]:


img


# In[29]:


folder = 'dataset/test/images2/'


# In[30]:

source = folder #+img['file_name']


#I = io.imread(source)
#plt.axis('off')
#plt.imshow(I)
#plt.show()


# In[31]:


result = detect_yolo_from_image(source)

print(result)
print(len(result))


# In[ ]:


#!pip freeze


# In[ ]:




