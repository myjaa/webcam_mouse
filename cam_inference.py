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


def load_image_into_numpy_array(path):
    print('SECOND WALA')
    """Load an image from file into a numpy array.
  
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
  
    Args:
      path: the file path to the image
  
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    print(image.size)
    (im_width, im_height) = image.size
    image = image.resize((im_width // 2, im_height // 2))
    print(image.size)
    (im_width, im_height) = image.size
    # image=np.resize(image,(int(im_height/1.5), int(im_width/1.5), 3))
    # print(image.shape)
    # image=np.array(image.getdata()).resize((int(im_height), int(im_width), 3))
    # (im_width, im_height) = image.size
    # return image.astype(np.uint8)
    # return np.reshape(image,(int(im_height/3), int(im_width/3), 3)).astype(np.uint8)
    return np.array(image.getdata()).reshape(
        (int(im_height), int(im_width), 3)).astype(np.uint8)


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

image_dir = 'fine_tuned_model'
img_names = os.listdir(image_dir)
print(img_names)
elapsed = []
for i in range(2):
    # image_path = os.path.join(image_dir, img_names[i])
    image_path = r'test.jpg'
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = np.expand_dims(image_np, 0)
    start_time = time.time()
    detections = detect_fn(input_tensor)

    # classes=detections['detection_classes'][0].numpy().astype(np.int32)
    # class_name=''
    # for j in range(detections['detection_boxes'][0].numpy().shape[0]):
    #   if classes[j] in six.viewkeys(category_index):
    #     class_name = category_index[classes[j]]['name']
    #   else:
    #     class_name = 'N/A'
    # print(class_name)

    end_time = time.time()
    elapsed.append(end_time - start_time)

    plt.rcParams['figure.figsize'] = [42, 21]
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.50,
        agnostic_mode=False)
    plt.subplot(2, 1, i + 1)
    plt.savefig('image.jpg')
    plt.imshow(image_np_with_detections)
    plt.show()

mean_elapsed = sum(elapsed) / float(len(elapsed))
print('Elapsed time: ' + str(mean_elapsed) + ' second per image')
