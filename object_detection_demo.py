import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image

import cv2
cap = cv2.VideoCapture(os.path.abspath(os.path.join(os.getcwd(),'movie_clips/james_bond_spectre.mp4')))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output/jamed_bond_spectre_detected.avi',fourcc,20.0,(848,360))

# This is needed since the next two modules are stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# because the model is based on coco images
NUM_CLASSES = 90

def select_and_download_model(MODEL_NAME='ssd_mobilenet_v1_coco_11_06_2017'):
  '''
  select a pre-trained model from 
  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
  This code uses a COCO-pretrained model. This implies that a number of target classes is 90
  A good alternative for a pre-trained model is MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'
  Pay attention to the trade-off of speed with accuracy. 
  '''
  
  MODEL_FILE = MODEL_NAME + '.tar.gz'
  DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

  # List of the strings that is used to add correct label for each box.
  PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

  # If this is the first time for the model of your choice, download it
  if not os.path.exists(MODEL_FILE):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

  # extract pb file 
  tar_file = tarfile.open(MODEL_FILE)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, os.getcwd())

  return PATH_TO_CKPT, PATH_TO_LABELS


def load_model(PATH_TO_CKPT,PATH_TO_LABELS):

  '''
  Load a (frozen) Tensorflow model into memory.
  '''

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  '''
  Loading label map
  Label maps map indices to category names.  
  Here we use internal utility functions from Tensorflow repo, 
  but anything that returns a dictionary mapping integers to appropriate string labels would be fine
  '''

  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  return detection_graph, category_index


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def detect_object(detection_graph,category_index):
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      while True:
        ret, image_np = cap.read()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        out.write(image_np)
        cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
        if cv2.waitKey(23) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break

# connect functions to run the whole cycle
def main():
  PATH_TO_CKPT, PATH_TO_LABELS = select_and_download_model()
  detection_graph, category_index = load_model(PATH_TO_CKPT,PATH_TO_LABELS)
  detect_object(detection_graph,category_index)

if __name__ == "__main__":
  main()