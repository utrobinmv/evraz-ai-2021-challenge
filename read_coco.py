from pycocotools.coco import COCO
print('aaa')
#import numpy as np
#import skimage.io as io
#import matplotlib.pyplot as plt
#import pylab
#pylab.rcParams['figure.figsize'] = (8.0, 10.0)

#dataDir='/path/to/your/coco_data'
#dataType='val2017'
annFile='my_coco.json' #.format(dataDir,dataType)
# Инициализировать API COCO для помеченных данных 
coco=COCO(annFile)

