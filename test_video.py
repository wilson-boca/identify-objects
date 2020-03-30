import os
import cv2
import numpy as np
import tensorflow as tf
from time import sleep
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.test.gpu_device_name():
   print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")
print(tf.test.is_gpu_available())

CWD_PATH = os.getcwd()
MODEL_NAME = 'bee_inference'
TEST_FOLDER = 'images/bees'
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'labelmap.pbtxt')
NUM_CLASSES = 1
# webcam = cv2.VideoCapture(0)
webcam = cv2.VideoCapture('http://192.168.0.22')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

x = 100
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = webcam.read()
            # image_np_expanded = np.expand_dims(image_np, axis=0)
            # image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # (boxes, scores, classes, num_detections) = sess.run(
            #     [boxes, scores, classes, num_detections],
            #     feed_dict={image_tensor: image_np_expanded})
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #     image_np,
            #     np.squeeze(boxes),
            #     np.squeeze(classes).astype(np.int32),
            #     np.squeeze(scores),
            #     category_index,
            #     use_normalized_coordinates=True,
            #     line_thickness=1)
            # cv2.imshow('Object detection', cv2.resize(image_np, (800, 600)))
            x += 1
            filename = '/home/rodrigo/projects/object/images/bees2/image' + str(int(x)) + ".jpeg";
            cv2.imwrite(filename, image_np)
            key = cv2.waitKey(1000)
            if key == 27:
                cv2.destroyAllWindows()
                webcam.release
                break
