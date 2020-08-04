import os
import scipy.misc
import numpy as np
import six
import time

from six import BytesIO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
import cv2

# Load the COCO Label Map
category_index = {
    1: {'id': 1, 'name': 'click'},
    2: {'id': 2, 'name': 'move'}}

start_time = time.time()
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load(r'fine_tuned_model\saved_model')
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time: ' + str(elapsed_time) + 's')

elapsed = []
#
# cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
vid = cv2.VideoCapture(0)
while True:
    success, frame = vid.read()
    if not success:
        print('ERROR')
        break

    frame = cv2.flip(frame, 1)
    input_tensor = np.expand_dims(frame, 0)
    start_time = time.time()
    detections = detect_fn(input_tensor)

    end_time = time.time()
    elapsed.append(end_time - start_time)

    # plt.rcParams['figure.figsize'] = [42, 21]
    label_id_offset = 1
    image_np_with_detections = frame.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1,
        min_score_thresh=.50,
        agnostic_mode=False)

    # print(image_np_with_detections.shape)

    # cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window",image_np_with_detections)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

mean_elapsed = sum(elapsed) / float(len(elapsed))
print('Elapsed time: ' + str(mean_elapsed) + ' second per image')
