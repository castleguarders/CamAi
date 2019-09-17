import sys
import os
import threading
from multiprocessing import Process
import queue
import datetime
import numpy as np
import cv2 as cv
import random
import logging

from . import CamAiMessage
#import CamAiMessage
#from . import CamAiConfig
from .CamAiConfig import Default_Objects_to_Watch

# Ratchet down tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#sys.path.append('./mrcnn')
#sys.path.append('./coco')
#import coco
import mrcnn_model as modellib
import mrcnn_config as mrconfig
import imutils

logger = logging.getLogger(__name__)
#logger.setLevel(logging.WARN)
#formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
#formatter = logging.Formatter('%(asctime)s:%(message)s')
formatter = logging.Formatter('%(asctime)s:%(name)s:%(funcName)s:%(lineno)d:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

#file_handler = logging.FileHandler('CamAiCameraWriter.errorlog')
#file_handler.setFormatter(formatter)
#file_handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.WARN)

#logger.addHandler(file_handler)
logger.addHandler(stream_handler)

DEBUG = False

Detector_Test_Image = "testimg.jpg"
Q_Depth_Profiler_Interval = 300

from enum import Enum


def get_model_path(modelfile):
        modulepath = os.path.abspath(__file__)
        moduledir = os.path.dirname(modulepath)
        modeldir = os.path.join(moduledir, "modeldata")
        modelfilepath = os.path.join(modeldir, modelfile)
        return modeldir, modelfilepath

class CamAiDetectionModels(Enum):
    mmaskrcnn = 100  # MaskRCNN from matterport
    kinceptionresnetv2 = 200  # From Keras
    kresnet50 = 201  # From Keras
    kresnext101 = 202  # From Keras
    kinceptionv3 = 203  # From Keras
    cvhaarcascadefrontal = 300  # From opencv
    cvhaarcascadeeye = 301  # From opencv
    cvdnnface = 302  # From opencv/caffe
    dlibcnnface = 303  # From dlib CNN version not HOG

# TODO:  This entire section is tied too closely to coco, need to decouple to handle other
# datasets

# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
coco_class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

def get_class_index(object_classname='person', dataset_type='coco'):
    if dataset_type is 'coco':
        if object_classname in coco_class_names:
            idx = coco_class_names.index(object_classname)
            return idx
        else:
            logger.warning("throwing value errror")
            raise ValueError("{} is not a known class name for {} dataset ".format(object_classname, dataset_type))
    else:
        logger.warning("throing value errror")
        raise ValueError("unknown dataset type: {} : ".format(dataset_type))

def get_class_name(object_index=0, dataset_type='coco'):
    if dataset_type is 'coco':
        if object_index > 0 and object_index < len(coco_class_names):
            return coco_class_names[object_index]
        else:
            logger.warning("throing index errror")
            raise IndexError("index: {} is out of bounds for {} dataset ".format(object_index, dataset_type))
    else:
        logger.warning("throing value errror")
        raise ValueError("unknown dataset type: {} : ".format(dataset_type))


Person_Classname = 'person'
Person_Index = coco_class_names.index(Person_Classname)
Car_Classname = 'car'
Car_Index = coco_class_names.index(Car_Classname)
Bicycle_Classname = 'bicycle'
Bicycle_Index = coco_class_names.index(Bicycle_Classname)
Motorcycle_Classname = 'motorcycle'
Motorcycle_Index = coco_class_names.index(Motorcycle_Classname)
Truck_Classname = 'truck'
Truck_Index = coco_class_names.index(Truck_Classname)
Bus_Classname = 'bus'
Bus_Index = coco_class_names.index(Bus_Classname)

Vehicle_Indexes = [Car_Index,
                   Bicycle_Index,
                   Motorcycle_Index,
                   Bus_Index,
                   Truck_Index
                   ]

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    # Apply mask to the image
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1-alpha) + alpha * c,
            image[:, :, n]
        )

    return image

# Returns the resized image, scale used for resizing, padding used _before_ resizing
# To reverse, resize by the inverse scale, then remove padding


def resize2SquareKeepingAspectRatio(img, size, interpolation=cv.INTER_AREA):
    h, w = img.shape[:2]
    # window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0), (0, 0)]

    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w:
        return cv.resize(img, (size, size), interpolation), scale, padding
    if h > w:
        dif = h
    else:
        dif = w

    scale = size/dif

    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)

    # bottom/left pad, followed by top/right pad
    padding = [
        (0, 0),
        (y_pos, x_pos),
        (h + y_pos, w + x_pos),
        (h + 2 * y_pos, w + 2 * x_pos)]

    # Grayscale image
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    # Color image
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv.resize(mask, (size, size), interpolation), scale, padding


def resize_image(image, min_dim=None, max_dim=None,
                 min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

# scale = 1
# scale = max_dim/ image_max
# return cv.resize(image, (round(h * scale), round(w * scale))
    # Resize image using bilinear interpolation
    if scale != 1:
        image = cv.resize(image, (round(h * scale), round(w * scale)))
        # image = resize(image, (round(h * scale), round(w * scale)),
        #               preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def load_test_image(image_path):
    # Load the specified image and return a [H,W,3] Numpy array.
    image = cv.imread(image_path, flags=cv.IMREAD_COLOR)

    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        logger.error(f"Unable to read a color image: {image_path}")

    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        logger.error(f"Image : {image_path} has an alpha channel")
        image = image[..., :3]

    return image


# Returning a dict instead of an array in case we have to add other return
# variables in the future
# Format of the dict is as follows
# {
# 'object_index0': {'matched_records': [{'class': c, 'score': sc, 'roi': roi},... N]
#                   'found': True, # Should always be true as object_index entries that weren't found aren't included
#                   'best_score': best_score,
#                   'best_match_index': best_index
#                  }
# ....
# 'object_indexN': {'matched_records': [{'class': c, 'score': sc, 'roi':  roi},... N]
#                   'found': True, # Should always be true as object_index entries that weren't found aren't included
#                   'best_score': best_score,
#                   'best_match_index': best_index
#                  }
# }

#print (Default_Objects_to_Watch)

def get_watched_matches(results, objects_of_interest=Default_Objects_to_Watch):
#def get_watched_matches(results, objects_of_interest):
    ids = results['class_ids']
    scores = results['scores']
    rois = results['rois']
    matched_objects_dict = {}

    # It's possible to get multiple entries with the same classid, e.g: 3
    # people detected, 4 cars detected
    for index, classid, score, roi in zip(range(len(ids)), ids, scores, rois):
        #logger.debug("Object of interest  {}".format(objects_of_interest))
        for object_classname in objects_of_interest:
            object_index = get_class_index(object_classname)
            logger.debug("Object of interest class {} , index {}".format(object_classname, object_index))
            found = False
            best_score = 0
            best_match_index = 0
            if (classid == object_index):
                logger.debug("Object of interest {} detected, score is {}".format(classid, scores))
                # Only return entries if threshold is met, otherwise false
                # alarms are being triggered
                if (score > objects_of_interest[object_classname]['detection_threshold']):
                    logger.debug(
                        "Object of interest detected over threshold probability {}".format(score))
                    if object_index not in matched_objects_dict:
                        matched_objects_dict[object_index] = {}
                        matched_objects_dict[object_index]['matched_records'] = []
                    found = True
                    matched_record = {'class': coco_class_names[classid],
                                      'score': score,
                                      'roi':   roi
                                      }
                    if score > best_score:
                        best_score = score
                        best_match_index = index
                    matched_objects_dict[object_index]['matched_records'].append(matched_record)
                    matched_objects_dict[object_index]['found'] = found
                    matched_objects_dict[object_index]['best_score'] = best_score
                    matched_objects_dict[object_index]['best_match_index'] = best_match_index
    logger.debug(
        "Returning matched objects dict  {}".format(matched_objects_dict))

    return matched_objects_dict


# @profile
def annotate_objects_in_frame(image, boxes, masks, ids, scores):
    names = coco_class_names
    num_instances = boxes.shape[0]

    if not num_instances:
        logger.debug("No instances to display!\n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    colors = random_colors(num_instances)
    height, width = image.shape[:2]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = "{} {:.2f}".format(label, score) if score else label
        logger.debug('    {}'.format(caption))
        image = cv.putText(
            image, caption, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image

# Input rectangle coordinates are expected to be in this format
# rect1 = [y1, x1, y2, x2], rect2 = [y3, x3, y4, x4]
# Returns the intersection over union of the two rectangles
def get_roi_iou(rect1, rect2):
    iou = 0

    # Index definitions
    y1 = 0
    x1 = 1
    y2 = 2
    x2 = 3

    if rect1 is None or rect2 is None:
        logger.warning("Null Rectangle passed Rect 1 is {} , Rect 2 is {}".format(rect1, rect2))
        return iou

    logger.debug("Rect 1 is {} , Rect 2 is {}".format(rect1, rect2))

    rightmost_start  = max(rect1[x1], rect2[x1])
    leftmost_end     = min(rect1[x2], rect2[x2])

    if (leftmost_end - rightmost_start) >= 0:
        x_overlap = leftmost_end - rightmost_start
    else:
        x_overlap = 0

    logger.debug ("rightmost_start: {}, leftmost_end: {}".format(rightmost_start, leftmost_end))
    logger.debug ("x_overlap: {}".format(x_overlap))

    highest_start  = max(rect1[y1], rect2[y1])
    lowest_end     = min(rect1[y2], rect2[y2])


    if (lowest_end - highest_start) >= 0:
        y_overlap = lowest_end - highest_start
    else:
        y_overlap = 0

    logger.debug ("highest_start: {}, lowest_end: {}".format(highest_start, lowest_end))
    logger.debug ("y_overlap: {}".format(y_overlap))

    overlap_area = x_overlap * y_overlap
    rect1_area =  (rect1[x2] - rect1[x1]) * (rect1[y2] - rect1[y1])
    logger.debug ("Overlap Area: {}".format(overlap_area))
    rect2_area =  (rect2[x2] - rect2[x1]) * (rect2[y2] - rect2[y1])
    intersection_area = rect1_area + rect2_area - overlap_area
    logger.debug ("Intersection Area: {}".format(intersection_area))

    if intersection_area is not 0:
        iou = overlap_area/intersection_area
    else:
        iou = 0

    logger.debug ("IOU: {}".format(iou))
    return iou

# Input rectangle coordinates are expected to be in this format
# rect1 = [y1, x1, y2, x2], rect2 = [y3, x3, y4, x4]
# Returns the direction of the move of rect1 to rect2 in radians and degrees
def get_move_direction(rect1, rect2):
    import math
    # Index definitions
    y1 = 0
    x1 = 1
    y2 = 2
    x2 = 3

    rect1_centroid_x = rect1[x1] + (rect1[x2] - rect1[x1])/2
    rect1_centroid_y = rect1[y1] + (rect1[y2] - rect1[y1])/2

    rect2_centroid_x = rect2[x1] + (rect2[x2] - rect2[x1])/2
    rect2_centroid_y = rect2[y1] + (rect2[y2] - rect2[y1])/2

    move_x = rect2_centroid_x - rect1_centroid_x
    move_y = rect2_centroid_y - rect1_centroid_y
    radianangle = math.atan2(move_y, move_x)
    degreeangle = math.degrees(radianangle)

    return radianangle, degreeangle


def get_objects_IncepResV2(detectimage, model=None, confidence_threshold=0.5):

    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.preprocessing import image
    from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

    # keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    # TODO: Need to figure out how to load weights from a non default local
    # file
    if model:
        dmodel = model
    else:
        dmodel = InceptionResNetV2(weights='imagenet')

    # TODO Get image size from modelconfig intead
    img, scale, padding = resize2SquareKeepingAspectRatio(
        detectimage, size=299)

    y = image.img_to_array(img)
    y = np.expand_dims(y, axis=0)
    y = preprocess_input(y)

    preds = dmodel.predict(y)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    results = []
    # should tie this to a threshold
    presults = decode_predictions(preds, top=20)[0]
    for predictedclass in presults:
        if predictedclass[2] >= confidence_threshold:
            results.append([predictedclass[1], predictedclass[2]])

    # print('Predicted:', decode_predictions(preds, top=3)[0])
    return results


def get_objects_Resnet50(detectimage, model=None, confidence_threshold=0.5):
    from keras.applications.resnet50 import ResNet50
    from keras.preprocessing import image
    from keras.applications.resnet50 import preprocess_input, decode_predictions

    if model:
        dmodel = model
    else:
        dmodel = ResNet50(weights='imagenet')

    # TODO Get image size from modelconfig intead
    img, scale, padding = resize2SquareKeepingAspectRatio(
        detectimage, size=224)

    y = image.img_to_array(img)
    y = np.expand_dims(y, axis=0)
    y = preprocess_input(y)

    preds = dmodel.predict(y)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    results = []
    # should tie this to a threshold
    presults = decode_predictions(preds, top=20)[0]
    for predictedclass in presults:
        if predictedclass[2] >= confidence_threshold:
            results.append([predictedclass[1], predictedclass[2]])

    # print('Predicted:', decode_predictions(preds, top=3)[0])
    return results


def get_objects_ResnetNeXt101(
        detectimage, model=None, confidence_threshold=0.5):
    from keras.applications.resnext import ResNeXt101
    from keras.preprocessing import image
    from keras.applications.resnext import preprocess_input, decode_predictions

    if model:
        dmodel = model
    else:
        dmodel = ResNeXt101(weights='imagenet')

    # TODO Get image size from modelconfig intead
    img, scale, padding = resize2SquareKeepingAspectRatio(
        detectimage, size=224)

    y = image.img_to_array(img)
    y = np.expand_dims(y, axis=0)
    y = preprocess_input(y)

    preds = dmodel.predict(y)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    results = []
    # should tie this to a threshold
    presults = decode_predictions(preds, top=20)[0]
    for predictedclass in presults:
        if predictedclass[2] >= confidence_threshold:
            results.append([predictedclass[1], predictedclass[2]])

    # print('Predicted:', decode_predictions(preds, top=3)[0])
    return results

# This is based on rather ancient haar cascades, doesn't really work for side profiles
# or occluded faces. Based on empirical tests, scale of 1.1 seems to work best
# an 1.0 scale has enormous false detect rate, and anything above 1.2
# doesn't detect much


def get_faces_haar(image, model=None, scaleFactor=1.1,
              minNeighbors=5, minSize=(30, 30), drawbox=False):
    if model is None:
        modeldir, frontalmodelfilepath = get_model_path("haarcascade_frontalface_default.xml")
        face_cascade = cv.CascadeClassifier(frontalmodelfilepath)
        #    'fdmodels/haarcascade_frontalface_default.xml')
        modeldir, eyemodelfilepath = get_model_path('haarcascade_eye.xml')
        #eye_cascade = cv.CascadeClassifier('fdmodels/haarcascade_eye.xml')
        eye_cascade = cv.CascadeClassifier(eyemodelfilepath)
    else:
        face_cascade = model['face_cascade']
        eye_cascade = model['eye_cascade']

    grayimage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
            grayimage,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize
     )

    eyes = eye_cascade.detectMultiScale(
            grayimage,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
     )

    # logger.debug("eyes is what kind of an object : {}".format(eyes))
    # logger.warn("cascade: Found {0} eyes".format(len(eyes)))

    # for (x, y, w, h) in eyes:
    #    both = cv.rectangle(face, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Which so called faces have eyes in them?
    # Coordinates are always in upper right quadrant, i.e. always positive
    faceimgs = []
    both = image
    if (len(eyes) > 0):
        faces_with_eyes = []
        eyes_in_faces = []
        for (fx, fy, fw, fh) in faces:
            for (ex, ey, ew, eh) in eyes:
                # Starting eye x coord overshoots face
                if ex > fx+fw:
                    logger.debug("x overshoot")
                # Ending eye x coord is undershoots face
                elif ex+ew < fx:
                    logger.debug("x undershoot")
                # Starting eye y coord overshoots face
                elif ey > fy+fh:
                    logger.debug("y overshoot")
                # Ending eye y coord undershoots face
                elif ey+eh < fy:
                    logger.debug("y undershoot")
                    pass
                else:
                    # There must be some overlap
                    faces_with_eyes.append((fx, fy, fw, fh))
                    eyes_in_faces.append((ex, ey, ew, eh))

        for ((fx, fy, fw, fh), (ex, ey, ew, eh)) in zip(
                faces_with_eyes, eyes_in_faces):
            if drawbox is True:
                both = cv.rectangle(both, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
                both = cv.rectangle(both, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            faceimg = image[fy:(fy+fh), fx:(fx+fw)]
            faceimgs.append(faceimg)
    else:
        for (x, y, w, h) in faces:
            if drawbox is True:
                both = cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            faceimg = image[y:(y+h), x:(x+w)]
            faceimgs.append(faceimg)

    # logger.debug("faces is what kind of an object : {}".format(faces))
    # logger.warn("cascade: Found {0} faces".format(len(faces)))

    return len(faces), faceimgs, len(eyes),  both


def get_faces_caffednn(
        image, model=None, scaleFactor=1.1, resizeimgto=(300, 300),
        confidence_threshold=0.5, drawbox=False):
    # Define paths
    modeldir, prototxt_path = get_model_path('deploy.prototxt')
    #prototxt_path = os.path.join('./fdmodels/deploy.prototxt')
    modeldir, caffemodel_path = get_model_path('weights.caffemodel')
    #caffemodel_path = os.path.join('./fdmodels/weights.caffemodel')

    if model:
        dmodel = model
    else:
        dmodel = cv.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    # Load image
    (h, w) = image.shape[:2]
    blob = cv.dnn.blobFromImage(
        image, scaleFactor, resizeimgto, (104.0, 117.0, 123.0))

    dmodel.setInput(blob)
    detections = dmodel.forward()

    # draw a bounding box around face
    max_confidence = 0.0
    num_faces = 0
    faceimgs = []
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]
        if confidence > max_confidence: max_confidence = confidence

        if (confidence > confidence_threshold):
            num_faces += 1
            text = "{:.2f}%".format(confidence * 100)
            logger.debug("Face met the threshold : {}".format(text))
            y = startY - 10 if startY - 10 > 10 else startY + 10
            if drawbox is True:
                cv.rectangle(image, (startX, startY), (endX, endY), (204, 0, 0), 2)
                cv.putText(
                    image, text, (startX, y),
                    cv.FONT_HERSHEY_SIMPLEX, 0.45, (204, 0, 0), 2)

            faceimg = image[startY:endY, startX:endX]
            faceimgs.append(faceimg)

    return num_faces, faceimgs, 0, image, max_confidence


def get_faces_dlibdnn(
        image, model=None, scaleFactor=1.1, resizeimgto=(300, 300),
        confidence_threshold=0.25, drawbox=False):
    import dlib
    # Define paths
    modeldir, model_data = get_model_path('mmod_human_face_detector.dat')
    #model_data = os.path.join('./fdmodels/mmod_human_face_detector.dat')

    # Read the model
    if model:
        dmodel = model
    else:
        dmodel = dlib.cnn_face_detection_model_v1(model_data)

    # TODO: This is a hack to get out out of cuda memory when dealing with large images,
    # Even at 11GB GPU memory, 4k images seem to fail
    h, w = image.shape[:2]
    if w > 1920:
        logger.warn("Resizing image from h:{}, w:{} ".format(h, w))
        image = imutils.resize(image, width=1920)
        h, w = image.shape[:2]
        logger.warn("Resized image to h:{}, w:{} ".format(h, w))

    max_confidence = 0.0

    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    # dets = cnn_face_detector(image, 1)
    dets = dmodel(image, 1)
    '''
    This detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
    These objects can be accessed by simply iterating over the mmod_rectangles object
    The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.

    It is also possible to pass a list of images to the detector.
        - like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)
    In this case it will return a mmod_rectangless object.
    This object behaves just like a list of lists and can be iterated over.
    '''
    num_faces = len(dets)
    faceimgs = []
    # logger.warn("dlib_cnn:Found {0} faces".format(num_faces))
    for i, d in enumerate(dets):
        logger.debug(
            "Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
                i,
                d.rect.left(),
                d.rect.top(),
                d.rect.right(),
                d.rect.bottom(),
                d.confidence))
        if d.confidence > max_confidence:
            max_confidence = d.confidence
        if (d.confidence >= confidence_threshold):
            if drawbox is True:
                image = cv.rectangle(image, (d.rect.left(), d.rect.top()),
                                     (d.rect.right(), d.rect.bottom()),
                                     (255, 0, 127), 2)
                text = format("{:.2f}%".format(d.confidence * 100))
                cv.putText(image, text, (d.rect.left(), d.rect.bottom()),
                           cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 127), 2)

            faceimg = image[d.rect.top():d.rect.bottom(), d.rect.left():d.rect.right()]
            faceimgs.append(faceimg)

    return num_faces, faceimgs, 0, image, max_confidence

# Check input list of images and associated crops of people, try to
# a) Find the best 'face' crop and return that for notifications
# b) Return a list of detected faces so it can be fed to the
#    face recognition step later on
def check_for_faces(images, cropped_images=None,
                    basedir="./", write_imgs=True, reduce_size=True,
                    reduce_size_to=(1080, 1920)
                    ):
    import dlib
    modeldir, dlibmodel_data = get_model_path('mmod_human_face_detector.dat')
    #dlibmodel_data = os.path.join('./fdmodels/mmod_human_face_detector.dat')
    dlibmodel = dlib.cnn_face_detection_model_v1(dlibmodel_data)

    modeldir, prototxt_path = get_model_path('deploy.prototxt')
    modeldir, caffemodel_path = get_model_path('weights.caffemodel')
    #prototxt_path = os.path.join('./fdmodels/deploy.prototxt')
    #caffemodel_path = os.path.join('./fdmodels/weights.caffemodel')
    caffemodel = cv.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    logger.warning("Detection : created dlib and caffe models")

    results = []
    frame_number = 0
    ts = datetime.datetime.now()
    camera_name = "camera_name_todo"
    max_index = 0
    max_composite_confidence = 0
    max_image  = None
    max_image1 = None
    max_image2 = None
    max_image3 = None

    all_faceimgs = []

    logger.warning("Detection : checking {} frames for faces".format(len(images)))
    for image, object_crops in zip(images, cropped_images):
        logger.debug("Detection : checking frame: {} for faces".format(frame_number))
        for object_crop in object_crops:
            if reduce_size is True:
                # Resize to a
                h, w = object_crop.shape[:2]
                if w > h and w > reduce_size_to[1]:
                    logger.warn("Resizing image from h:{}, w:{} ".format(h, w))
                    object_crop = imutils.resize(object_crop, width=reduce_size_to[1])
                    h, w = object_crop.shape[:2]
                    logger.warn("Resized image to h:{}, w:{} ".format(h, w))
                elif h > reduce_size_to[0]:
                    logger.warn("Resizing image from h:{}, w:{} ".format(h, w))
                    object_crop = imutils.resize(object_crop, height=reduce_size_to[0])
                    h, w = object_crop.shape[:2]
                    logger.warn("Resized image to h:{}, w:{} ".format(h, w))
                else:
                    pass

            # Ensemble detection from hell, wait for keras 2.2.5 for resnext
            # resnext_results = CamAiDetection.get_objects_ResnetNeXt101(image)
            # incepresv2_results = CamAiDetection.get_objects_IncepResV2 (image)
            # resnet50_results = CamAiDetection.get_objects_Resnet50 (image)
            # logger.warn("Resnext detected: {}".format(resnext_results))
            # logger.warn("Incepresv2 detected: {}".format(incepresv2_results))
            # logger.warn("Resnet50 detected: {}".format(resnet50_results))

            # Try to do face detection to weed out false
            # positives
            numfaces1, faceimgs, numeyes1, faceimage1 = get_faces_haar(image=object_crop, minNeighbors=10)
            if numfaces1 > 0 and numeyes1 > 1:
                all_faceimgs.extend(faceimgs)
                if write_imgs is True:
                    alarm_image_file = basedir + camera_name \
                        + "_alarm_faces_haar_frame_" + str(frame_number) + str(ts) + ".png"
                    rc = cv.imwrite(alarm_image_file, faceimage1)

            numfaces2, faceimgs, numeyes2, faceimage2, maxconf2 = get_faces_caffednn(
                image=object_crop, model=caffemodel)
            if numfaces2 > 0:
                all_faceimgs.extend(faceimgs)
                if write_imgs is True:
                    alarm_image_file = basedir + camera_name \
                        + "_alarm_faces_caffednn_frame_" + str(frame_number) + str(ts) + ".png"
                    rc = cv.imwrite(alarm_image_file, faceimage2)

            numfaces3, faceimgs, numeyes2, faceimage3, maxconf3 = get_faces_dlibdnn(
                image=object_crop, model=dlibmodel)
            if numfaces3 > 0:
                all_faceimgs.extend(faceimgs)
                if write_imgs is True:
                    alarm_image_file = basedir + camera_name \
                        + "_alarm_faces_dlibdnn_frame_" + str(frame_number) + str(ts) + ".png"
                    rc = cv.imwrite(alarm_image_file, faceimage3)

            results.append([numfaces1, numfaces2, numfaces3])

            composite_confidence = min(numfaces1, numeyes1, 0.3) + maxconf2*min(numfaces2, 1) + maxconf3*min(numfaces3,1)
            if composite_confidence > max_composite_confidence:
                max_composite_confidence = composite_confidence
                max_index = len(results) - 1
                max_image1 = faceimage1
                max_image2 = faceimage2
                max_image3 = faceimage3
                max_image = image

            frame_number += 1

    if max_composite_confidence !=0 and write_imgs is True:
        logger.warning("Max face match parameters are {} : ".format(results[max_index]))
        alarm_image_file = basedir + camera_name \
            + "_alarm_faces_haar_frame_MAX" + str(frame_number) + str(ts) + ".png"
        rc = cv.imwrite(alarm_image_file, max_image1)

        alarm_image_file = basedir + camera_name \
            + "_alarm_faces_caffednn_frame_MAX" + str(frame_number) + str(ts) + ".png"
        rc = cv.imwrite(alarm_image_file, max_image2)

        alarm_image_file = basedir + camera_name \
            + "_alarm_faces_dlibdnn_frame_MAX" + str(frame_number) + str(ts) + ".png"
        rc = cv.imwrite(alarm_image_file, max_image3)

    # In case no faces were found, just pick the first frame
    if max_image is None:
        max_image = images[0]

    return results, max_image, max_index, images, all_faceimgs

class CamAiDetection (object):

    def __init__(
                self,
                detect_queues,
                response_queues,
                name='Camera Name',
                pipeline_image_resize=True,
                multiprocessing=False,
                singledetectorqueue=True
                ):

        self.detect_queues = detect_queues
        self.response_queues = response_queues
        self.pipeline_image_resize = pipeline_image_resize
        self.name = name
        self.config = self.get_detection_config()
        self.multiprocessing = multiprocessing
        self.singledetectorqueue = singledetectorqueue

    def get_detection_config(self):
        # TODO: Assuming maskrcnn is the hard coded detection engine, this
        # should be config based
        if self.pipeline_image_resize is True:
            mode = "none"
        else:
            mode = "square"

        #class InferenceConfig(coco.CocoConfig):
        class InferenceConfig(mrconfig.CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            # We will do the resizing before sending for detection
            IMAGE_RESIZE_MODE = mode

        config = InferenceConfig()
        # config.display()

        return config


    def get_detection_model(self):
        # modulepath = os.path.abspath(__file__)
        # ROOT_DIR = os.path.dirname(modulepath)
        # MODEL_DIR = os.path.join(ROOT_DIR, "modeldata")
        # COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")
        modeldir, cocomodelfilepath = get_model_path("mask_rcnn_coco.h5")

        try:
            model = modellib.MaskRCNN(
                mode='inference',
                model_dir=modeldir,
                config=self.config)
        # TODO: Find out the specific exceptions that could be thrown and
        # handle them specifically
        except Exception as e:
            logger.warn("Model init failed, should exit: {}".format(e))
            return None

        try:
            model.load_weights(cocomodelfilepath, by_name=True)
        except ImportError:
            logger.warn("Model loading weights failed, should exit")
            return None

        return model

    def resize_image(self, image):
        if self.pipeline_image_resize is True:
            return resize2SquareKeepingAspectRatio(
                image, size=self.config.IMAGE_MAX_DIM,
                interpolation=cv.INTER_AREA)
        else:
            return image, 1, [(0, 0), (0, 0), (0, 0), (0, 0)]

    # TODO Make functional
    def run_as_process(self):
        if (self.multiprocessing is True):
            return True
        else:
            return False

    def start(self):
        if self.run_as_process():
            self.detection = Process(target=self._object_detector_server,
                                    args=([]),
                                    name="detection")

        else:
            self.detection = threading.Thread(
                target=self._object_detector_server, args=(
                    []), name="detection")
            self.detection.do_detect = True
        logger.warn("Starting detector")
        self.detection.start()

    def stop(self):
        self.detection.do_detect = False
        logger.warn("Stopping detector")

    def join(self, waittime=10):
        self.detection.join(waittime)
        logger.warn("Join detector")


# Using this to run detector in multiple processes
# @profile
def object_detector_server(self, camera_names):

    camera_detect_queues = self.detect_queues
    camera_response_queues = self.response_queues

    model = self.get_detection_model()
    if model is None:
        logger.warn("Could not initialize detection model, exiting")
        return

    # Prime the model fully and test a sample image to ensure
    # all libraries are preloaded before reader threads
    # Otherwise we are overrunning buffers at init with sometimes irreversible
    # decoding issues
    modeldir, testimagepath = get_model_path(Detector_Test_Image)
    testimage = load_test_image(testimagepath)
    if np.any(testimage):
        molded_testimage, scale, padding = self.resize_image(testimage)
        height, width = molded_testimage.shape[:2]
        logger.debug(
            "image height {} width {}, length of image array {}, batch_size {}".format(
                height,
                width,
                len(molded_testimage),
                model.config.BATCH_SIZE))
        testresults = model.detect([molded_testimage], verbose=0)
        logger.debug(
            "Detector results for test image: {} results {}".format(
                Detector_Test_Image, testresults))
    else:
        logger.warn(
            "Testing with image {} failed, detector is exiting".format(
                Detector_Test_Image))
        return

    detector = threading.currentThread()
    results = [None]*len(camera_detect_queues)

    do_detect = True  # Process mode get keyboard exits
    if (self.singledetectorqueue is True):
        waitstyle = True # Significantly more efficient per tests
    else:
        waitstyle = False

    while do_detect is True:
        if (self.multiprocessing is False):
            # TODO: In progress exit notifications only through queues
            # do_detect = detector.do_detect
            pass
        try:
            # Note: In case of single queue, all elements are going to be the
            # same queue, so no harm in looping, TODO: relook to get rid of
            # redundancy in queue creation time
            for index, name in enumerate(camera_names):
                try:
                    message = camera_detect_queues[index].get(waitstyle)
                    if message.msgtype == CamAiMessage.CamAiMsgType.detectimage:
                        camera_handle = message.msgdata['camera_handle']
                        images = message.msgdata['image']
                        #logger.debug(
                        #    "Got a frame from camera with camera_handle: {}, ".format(camera_handle))
                        results[camera_handle] = model.detect(images, verbose=0)
                        try:
                            logger.debug(
                                "Sending response to frame from camera {} using handle {}, \
                                     index is  {}".format(detector.name, camera_handle, index))
                            resultmessage = CamAiMessage.CamAiDetectResult(
                                camera_handle, results[camera_handle])
                            camera_response_queues[camera_handle].put(
                                resultmessage)
                        except queue.Full:
                            logger.warn(
                                "response queue is full for camera: {}".format(index))
                    elif message.msgtype == CamAiMessage.CamAiMsgType.quit:
                        logger.warn(
                            "Detector: {} Got a quit message, Stopping detection".format(detector.name))
                        do_detect = False
                        break
                    else:
                        logger.warn(
                            "Unknown message type received, debug : {}".format(detector.name))

                except queue.Empty:
                    #logger.debug("queue is empty for camera")
                    pass

        except KeyboardInterrupt:
            break

    # Start Cleaning up all child threads/processes here
    for index, name in enumerate(camera_names):
        # Tell threads/processes to not wait for any more queue items, Observer
        # most likely already got the message from parent already,
        quitmessage = CamAiMessage.CamAiQuitMsg()
        camera_response_queues[index].put(quitmessage)

    logger.warn(
        "Detector: {} Returning".format(detector.name))
    return



# Unit Tests
# Instantiation fake test
def class_init_test():
    detection_queues = {
                   'detect_queues': 'camera_detect_queues',
                   'response_queues': 'camera_response_queues',
           }
    mc = CamAiDetection(1, detection_queues, 'hellocam')

    print("Camera is {}".format(mc))
    image = cv.imread("./camface.png")
    numfaces, faceimgs, numeyes, imageface = get_faces_haar(image)
    # cv.imwrite('./personface.jpg', imageface)
    cv.imwrite('./personfacewitheyes.jpg', imageface)


# Test Face detection with OpenCV Haar Cascades for Face
def cv2_haar_face_test():
    rootdir = './videos'
    extensions = ('.jpg', '.png', '.gif')
    annotationPrefix = "annotated_"

    scaleFactors = [1.2, 1.3, 1.5, 1.8]
    # scaleFactors = [2.0]
    for scaleFactor in scaleFactors:
        logger.warn("Trying scaleFactor: {}".format(scaleFactor))
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext in extensions:
                    if annotationPrefix in file:
                        logger.warn(
                            "Previously created file found: {}. Skipping.".format(file))
                        continue
                    filewithpath = os.path.join(subdir, file)
                    logger.warn(filewithpath)
                    image = cv.imread(filewithpath)
                    numfaces, faceimgs, numeyes, imageface = get_faces_haar(
                        image, scaleFactor)
                    annotatedfile = os.path.join(
                        subdir, (annotationPrefix + str(scaleFactor) + file))
                    logger.warn("writing to {}".format(annotatedfile))
                    cv.imwrite(annotatedfile, imageface)

# Test Face detection with OpenCV Caffe DNN Face Detector


def cv2_dnn_face_test():
    rootdir = './videos'
    extensions = ('.jpg', '.png', '.gif')
    annotationPrefix = "annotated_"

    # scaleFactors = [1.2, 1.3, 1.5, 1.8]
    scaleFactors = [1.0, 1.0, 1.0]
    confidences = [0.5, 0.4, 0.25]
    for scaleFactor, confidence in zip(scaleFactors, confidences):
        logger.warn("Trying scaleFactor: {}".format(scaleFactor))
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext in extensions:
                    if annotationPrefix in file:
                        logger.warn(
                            "Previously created file found: {}. Skipping.".format(file))
                        continue
                    filewithpath = os.path.join(subdir, file)
                    logger.warn(filewithpath)
                    image = cv.imread(filewithpath)
                    # resizeimgto=(300,300), confidence_threshold=0.5
                    numfaces, faceimgs, numeyes, imageface = get_faces_caffednn(
                        image, scaleFactor, (2048, 2048), confidence)
                    annotatedfile = os.path.join(
                        subdir, (annotationPrefix + str(scaleFactor) + '_' + str(confidence) + file))
                    if numfaces > 0:
                        logger.warn(
                            "writing to {}, scaleFactor: {}, confidence threshold: \
                                {}".format(
                                annotatedfile, scaleFactor, confidence))
                        cv.imwrite(annotatedfile, imageface)

# Test Face detection with Dlib CNN Face Detector


def dlib_cnn_face_test():
    rootdir = './videos'
    extensions = ('.jpg', '.png', '.gif')
    annotationPrefix = "annotated_"

    scaleFactors = [1.0, 1.0, 1.0]
    confidences = [0.5, 0.4, 0.25]
    for scaleFactor, confidence in zip(scaleFactors, confidences):
        logger.warn(
            "Trying scaleFactor: {}, confidence:{}".format(
                scaleFactor, confidence))
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext in extensions:
                    if annotationPrefix in file:
                        logger.warn(
                            "Previously created file found: {}. Skipping.".format(file))
                        continue
                    filewithpath = os.path.join(subdir, file)
                    logger.warn(filewithpath)
                    image = cv.imread(filewithpath)
                    numfaces, faceimgs, numeyes, imageface = get_faces_dlibdnn(
                        image, scaleFactor, (2048, 2048), confidence)
                    annotatedfile = os.path.join(
                        subdir, (annotationPrefix + str(scaleFactor) + '_' + str(confidence) + file))
                    if numfaces > 0:
                        logger.warn(
                            "writing to {}, scaleFactor: {}, confidence threshold: \
                                {}".format(
                                annotatedfile, scaleFactor, confidence))
                        cv.imwrite(annotatedfile, imageface)


# Extract faces from videos so they can be used for training/recognition
# facedir will be treated as a subdirectory of basedir
def extract_faces(basedir='./videos', facedir='faces_dlibcnn'):
    import dlib
    extensions = ('.mp4', '.mkv')
    facePrefix = "face_"
    model_data = os.path.join('./fdmodels/mmod_human_face_detector.dat')
    model = dlib.cnn_face_detection_model_v1(model_data)

    scaleFactors = [1.0, ]  # Dlib doesn't have a scaleFactor
    # Dlib CNN seems to err towards accuracy, i.e. few false positives
    confidences = [0.25, ]
    for scaleFactor, confidence in zip(scaleFactors, confidences):
        logger.warn(
            "Trying scaleFactor: {}, confidence:{}".format(
                scaleFactor, confidence))
        for subdir, dirs, files in os.walk(basedir):
            for file in files:
                fileprefix, ext = os.path.splitext(file)
                # ext = os.path.splitext(file)[-1].lower()
                ext = ext.lower()
                if ext in extensions:
                    filewithpath = os.path.join(subdir, file)
                    logger.warn("reading video: {}".format(filewithpath))
                    # Read the video file
                    facecounter = 0
                    facevideo = cv.VideoCapture(filewithpath)
                    while facevideo.isOpened():
                        ret, image = facevideo.read()
                        if ret is False:
                            logger.warn(
                                "Got a read error, must be EOF, frame count {}".format(facecounter))
                            pass
                        facecounter += 1
                        if np.any(image):
                            numfaces, faceimgs, numeyes, faceimage = get_faces_dlibdnn(
                                image, model, scaleFactor, (2048, 2048), confidence)
                            logger.warn("Subdir is {}".format(subdir))
                            facefile = os.path.join(
                                subdir,
                                ("./" + facedir + "/" + facePrefix +
                                 str(facecounter) + str(scaleFactor) + "_" +
                                 str(confidence) + fileprefix + ".png"))
                            if numfaces > 0:
                                logger.warn(
                                    "writing to {}, scaleFactor: {}, confidence threshold: {}\
                                        ".format(
                                        facefile, scaleFactor, confidence))
                                cv.imwrite(facefile, faceimage)
                            pass
                        else:
                            logger.warn("No Image")
                            break

# Test object detection with InceptionResnetv2


def inception_resnet_test(rootdir=None):
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.preprocessing import image
    from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

    if rootdir is None:
        rootdir = './videos/faces'
    extensions = ('.jpg', '.png', '.gif')
    annotationPrefix = "annotated_"

    dmodel = InceptionResNetV2(weights='imagenet')

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in extensions:
                if annotationPrefix in file:
                    logger.warn(
                        "Previously created file found: {}. Skipping.".format(file))
                    continue
                filewithpath = os.path.join(subdir, file)
                logger.warn(filewithpath)
                img = cv.imread(filewithpath)
                results = get_objects_IncepResV2(img, dmodel)
                logger.warn("Results: {}".format(results))

# Test object detection with ResNet50


def resnet50_test(rootdir=None):
    from keras.applications.resnet50 import ResNet50
    from keras.preprocessing import image
    from keras.applications.resnet50 import preprocess_input, decode_predictions

    if rootdir is None:
        rootdir = './videos/faces'
    extensions = ('.jpg', '.png', '.gif')
    annotationPrefix = "annotated_"

    dmodel = ResNet50(weights='imagenet')

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in extensions:
                if annotationPrefix in file:
                    logger.warn(
                        "Previously created file found: {}. Skipping.".format(file))
                    continue
                filewithpath = os.path.join(subdir, file)
                logger.warn(filewithpath)
                img = cv.imread(filewithpath)
                results = get_objects_Resnet50(img, dmodel)
                logger.warn("Results: {}".format(results))

# Test object detection with ResNet50


def resnext_test(rootdir=None):
    # keras.applications.resnext.ResNeXt101(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    from keras.applications.keras_applications.resnext import ResNeXt101
    from keras.preprocessing import image
    from keras.applications.keras_applications.resnext import preprocess_input, decode_predictions

    if rootdir is None:
        rootdir = './videos/faces'
    extensions = ('.jpg', '.png', '.gif')
    annotationPrefix = "annotated_"

    dmodel = ResNeXt101(weights='imagenet')

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in extensions:
                if annotationPrefix in file:
                    logger.warn(
                        "Previously created file found: {}. Skipping.".format(file))
                    continue
                filewithpath = os.path.join(subdir, file)
                logger.warn(filewithpath)
                img = cv.imread(filewithpath)
                results = get_objects_Resnet50(img, dmodel)
                logger.warn("Results: {}".format(results))

        dmodel = ResNeXt101(weights='imagenet')


if __name__ == '__main__':
    # class_init_test()
    # cv2_haar_face_test()
    # cv2_dnn_face_test()
    # dlib_cnn_face_test()
    # inception_resnet_test('./videos/personcrops')
    resnet50_test('./videos/personcrops')
    # resnext_test()
    # extract_faces('./videos/facetest', 'faces_dlibcnn')
