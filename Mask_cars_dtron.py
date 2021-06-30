import glob
import numpy
import torch, torchvision
from torch._C import Size
from tqdm import tqdm
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
import statistics

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def check_object_size_ratio(pred_masks):
    black_pixels = cv2.countNonZero(pred_masks)
    total_pixels = pred_masks.shape[0] * pred_masks.shape[1]
    car_pixels = total_pixels-black_pixels
    size_ratio = car_pixels/ total_pixels
    return size_ratio

img_dir = "./dataset/A2MAC1_frontal_masked/"
mask_dir = img_dir + "masked/"
non_car_images = []
ratio = []
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

images = sorted(glob.glob(img_dir + '*.jpg'), key=len)
for image in tqdm(images):
    print(image)
    im = cv2.imread(image)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    #print(outputs["instances"])
    #check stepbystep: car, then for bus then for truck 
    i=[(outputs["instances"].pred_classes==2).nonzero(as_tuple=True)[0]]
    if i[0].nelement()==0:
        print("not a car")
        i=[(outputs["instances"].pred_classes==5).nonzero(as_tuple=True)[0]]
        if i[0].nelement()==0:
            print("not a bus")
            i=[(outputs["instances"].pred_classes==7).nonzero(as_tuple=True)[0]]
            if i[0].nelement()==0:
                print("not a truck, next image")
                non_car_images.append(image)
                continue
    print(outputs["instances"].pred_classes)
    print(i)
    areas=outputs["instances"][i].pred_boxes.area()
    print(areas)
    j=torch.argmax(areas).item()
    j=i[0].cpu().numpy()[j]
    print(j)
    pred_masks=outputs["instances"][int(j)].pred_masks.cpu().squeeze(0).long().numpy()
    pred_masks_ch1 = pred_masks.copy()
    pred_masks=numpy.dstack((pred_masks,pred_masks, pred_masks))
    pred_masks=pred_masks*255
    pred_masks=pred_masks.astype(np.uint8)
    masked_image=cv2.bitwise_and(pred_masks, im)
    if check_object_size_ratio(pred_masks_ch1) > 0.8:
        non_car_images.append(image)
        continue
    cv2.imwrite(mask_dir +  os.path.basename(image) ,masked_image)
    print(check_object_size_ratio(pred_masks_ch1))
    ratio.append(check_object_size_ratio(pred_masks_ch1))
    
with open(mask_dir + 'non_car_imgs.txt', 'w') as f:
    for item in non_car_images:
        f.write("%s\n" % item)



    
    
    
    