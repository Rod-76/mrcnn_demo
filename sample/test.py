import cv2
import model as modellib
import sys
sys.path.append("C:\\Users\\jemar\\Desktop\\Python Projects\\OpenCV\\Fin\\mrcnn_demo")
from custom import *

annotations_path = "annotations.json"

dataset_train = load_image_dataset(os.path.join("C:\\Users\\jemar\\Desktop\\Python Projects\\OpenCV\\Fin\\mrcnn_demo\\", annotations_path), "C:\\Users\\jemar\\Desktop\\Python Projects\\OpenCV\\Fin\\mrcnn_demo\\dataset", "train")
dataset_val = load_image_dataset(os.path.join("C:\\Users\\jemar\\Desktop\\Python Projects\\OpenCV\\Fin\\mrcnn_demo\\", annotations_path), "C:\\Users\\jemar\\Desktop\\Python Projects\\OpenCV\\Fin\\mrcnn_demo\\dataset", "val")
class_number = dataset_train.count_classes()
print('Train: %d' % len(dataset_train.image_ids))
print('Validation: %d' % len(dataset_val.image_ids))
print("Classes: {}".format(class_number))

# Load Test Model in inference mode
model_path = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
test_model, inference_config = load_inference_model(class_number, model_path)
