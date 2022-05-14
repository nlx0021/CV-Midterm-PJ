import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from yolo import YOLO
from utils.utils import get_classes
from utils.utils_map_train import get_coco_map, get_map

import random
from shutil import rmtree


classes_path    = 'model_data/voc_classes.txt'
VOCdevkit_path  = 'VOCdevkit'
map_out_path    = 'map_out'

def get_criterion_val():
    MINOVERLAP      = 0.5
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/val.txt")).read().strip().split()
    random.seed(21)
    random.shuffle(image_ids)
    image_ids = image_ids[:500]
    
    if os.path.exists(map_out_path):
        rmtree(map_out_path)
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))
    if not os.path.exists(os.path.join(map_out_path, 'results')):
        os.makedirs(os.path.join(map_out_path, 'results'))
        
    class_names, _ = get_classes(classes_path)
    
    # Load model
    yolo = YOLO(confidence = 0.001, nms_iou = 0.5)
    
    # Get predict result
    count = 0
    for image_id in image_ids:
        if count % 100 == 0:
            print(count)
        image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
        image       = Image.open(image_path)
        yolo.get_map_txt(image_id, image, class_names, map_out_path)
        count += 1
        
    # Get ground truth
    for image_id in image_ids:
        with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
            root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult')!=None:
                    difficult = obj.find('difficult').text
                    if int(difficult)==1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bndbox  = obj.find('bndbox')
                left    = bndbox.find('xmin').text
                top     = bndbox.find('ymin').text
                right   = bndbox.find('xmax').text
                bottom  = bndbox.find('ymax').text

                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                    
    # Get map          
    mAP, mIOU, mIOU_mod, acc = get_map(MINOVERLAP, False, path = map_out_path)
    
    return mAP, mIOU, mIOU_mod, acc