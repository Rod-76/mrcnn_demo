import cv2
import os
from m_rcnn import *
import model as modellib
import skimage.io
import visualize

annotations_path = "annotations.json"
dataset_train = load_image_dataset(os.path.join("C:\\Users\\jemar\\Desktop\\Python Projects\\OpenCV\\Fin\\mrcnn_demo\\", annotations_path), "C:\\Users\\jemar\\Desktop\\Python Projects\\OpenCV\\Fin\\mrcnn_demo\\dataset", "train")
dataset_val = load_image_dataset(os.path.join("C:\\Users\\jemar\\Desktop\\Python Projects\\OpenCV\\Fin\\mrcnn_demo\\", annotations_path), "C:\\Users\\jemar\\Desktop\\Python Projects\\OpenCV\\Fin\\mrcnn_demo\\dataset", "val")
class_number = dataset_train.count_classes()
print('Train: %d' % len(dataset_train.image_ids))
print('Validation: %d' % len(dataset_val.image_ids))
print("Classes: {}".format(class_number))


# Load Test Model in inference mode
model_path = ("C:\\Users\\jemar\\Desktop\\Python Projects\\OpenCV\\Fin\\mrcnn_demo\\white_oyster_mask_rcnn.h5")
test_model, inference_config = load_inference_model(class_number, model_path)

# Test on a random image
test_random_image(test_model, dataset_val, inference_config)