import sys
import os
import numpy as np
import cv2 as cv
import logging
import dlib
from math import sqrt

#import CamAiMessage
from . import CamAiMessage

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
#DEBUG = True

# TODO: Find if it's possible to use native dlib methods in python in the
# future versions
def euclidean_dist(vector_x, vector_y):
    if len(vector_x) != len(vector_y):
        raise Exception('Vectors must be same dimensions')
    return sum((vector_x[dim] - vector_y[dim]) ** 2 for dim in range(len(vector_x)))

def get_model_path(modelfile):
        modulepath = os.path.abspath(__file__)
        moduledir = os.path.dirname(modulepath)
        modeldir = os.path.join(moduledir, "modeldata")
        modelfilepath = os.path.join(modeldir, modelfile)
        return modeldir, modelfilepath


class CamAiDetectionFaces (object):

    def __init__(
        self,
        #modeldir= None,
        knownfacesdir='./known/people'):

        modeldir, predictor_path = get_model_path('shape_predictor_5_face_landmarks.dat')
        modeldir, face_rec_model_path = get_model_path('dlib_face_recognition_resnet_model_v1.dat')

        self.name = "CamAiDetectionFaces: "
        self.init_face_recognition_models(predictor_path, face_rec_model_path)
        self.init_known_face_descriptors(knownfacesdir=knownfacesdir)


    def init_face_recognition_models(
        self,
        predictor_path='./modeldata/shape_predictor_5_face_landmarks.dat',
        face_rec_model_path='./modeldata/dlib_face_recognition_resnet_model_v1.dat'):

        # Only init once
        if hasattr(self, 'face_detector') is False:
            try:
                # Load all the models we need: a face_detector to find the faces, a shape predictor
                # to find face landmarks so we can precisely localize the face, and finally the
                # face recognition model.
                self.face_detector = dlib.get_frontal_face_detector()
                self.sp = dlib.shape_predictor(predictor_path)
                self.facerec = dlib.face_recognition_model_v1(face_rec_model_path)
                logger.debug("Face Recognition Model inited")

            except Exception as e:
                logger.exception("Face Recognition Model init failed, should exit")


    # Returns a dictionary with key = name of person, value = list of face
    # descriptors (could be from one or more input images)
    # TODO: Evaluate comparing against a mean instead of many face descriptors for
    # the same person, investigate jittering each image
    def init_known_face_descriptors(self, knownfacesdir='./known/people'):
        if knownfacesdir is None:
            knownfacesdir = './known/people'
        extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

        face_detector = self.face_detector
        sp = self.sp
        facerec = self.facerec

        face_dict = {}
        for subdir, dirs, files in os.walk(knownfacesdir):
            if len(dirs) != 0:
                continue

            logger.debug("Subdir is {}, dirs is : {}, files are: {}".format(subdir, dirs, files))
            person_name = os.path.basename(subdir)
            person_face_descriptors = []

            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext in extensions:
                    filewithpath = os.path.join(subdir, file)
                    logger.debug(filewithpath)
                    img = cv.imread(filewithpath)
                    #jittered_imgs = dlib.jitter_image(img, num_jitters=10, disturb_colors=True)
                    faces = face_detector(img, 0)
                    logger.debug("Number of faces in image {} is {}".format(filewithpath, len(faces)))
                    for k, d in enumerate(faces):
                        shape = sp(img, d)
                        # Compute the 128D vector describing the face in img identified by shape.
                        face_descriptor = facerec.compute_face_descriptor(img, shape)
                        person_face_descriptors.append(face_descriptor)

            logger.warning("Number of face descriptors for {} is {}".format(person_name, len(person_face_descriptors)))
            face_dict[person_name] = person_face_descriptors

        self.face_dict = face_dict

    # Find the closest match to people in face_dict, it will return the best
    # match (if any found) for each face in the input image, the second return
    # value tells the caller the number of unmatched faces
    def find_face_matches(self, img, threshold=0.6, scaling=0):
        face_detector = self.face_detector
        sp = self.sp
        facerec = self.facerec
        face_dict = self.face_dict

        faces = face_detector(img, scaling)
        num_faces = len(faces)
        logger.debug("Number of faces detected: {} ".format(len(faces)))
        matches = {}

        if DEBUG is True:
            win = dlib.image_window()
            win.clear_overlay()
            win.set_image(img)

        # Now process each face we found.
        face_triple = []
        for index, d in enumerate(faces):
            face_triple.append([])
            facename = 'face_' + str(index)
            # Get the landmarks/parts for the face in box d
            shape = sp(img, d)

            # Compute the 128D vector that describes the face in img identified by
            # shape.
            face_descriptor = facerec.compute_face_descriptor(img, shape)

            for person in face_dict:
                logger.debug("Comparing against {}".format(person))
                if DEBUG is True:
                    # Draw the face landmarks on the screen so we can see what face is currently being processed.
                    win.clear_overlay()
                    win.add_overlay(d)
                    win.add_overlay(shape)
                    dlib.hit_enter_to_continue()
                for fd in face_dict[person]:
                    face_distance = euclidean_dist(face_descriptor, fd)
                    logger.debug("Euclidean distance from {} is {}".format(person, face_distance))

                distance_tuple = [person, face_distance]
                face_triple[index].append(distance_tuple)
                logger.debug("Best euclidean distance from {} is {}".format(person, face_distance))

        for facenum, face_tuples in enumerate(face_triple):
            logger.debug("Face tuples for face {}: {}".format(facenum, face_tuples))
            best_match = 1
            best_person_match = ''
            no_match = True
            for matchedperson in face_tuples:
                if (best_match > matchedperson[1]) and (matchedperson[1] <= threshold):
                    matches[facenum] =  matchedperson
                    best_match = matchedperson[1]

        unmatched = num_faces - len(matches)
        logger.debug ("Matched {} faces, could not match {} faces".format(len(matches), unmatched))
        return num_faces, matches, unmatched

    # Returns the number of matching faces between the two images
    # 0 if no matches, 1 or more if matches are found
    def compare_faces(self, img1, img2, threshold=0.6, scaling=0):
        face_detector = self.face_detector
        sp = self.sp
        facerec = self.facerec

        dets1 = face_detector(img1, scaling)
        dets2 = face_detector(img2, scaling)
        logger.debug("Number of faces detected img1: {}, img2: {}".format(len(dets1), len(dets2)))

        matchesfound=0
        # Now process each face we found.
        for k1, d1 in enumerate(dets1):
            # Get the landmarks/parts for the face in box d1.
            shape1 = sp(img1, d1)

            # Compute the 128D vector that describes the face in img identified by
            # shape.
            face_descriptor1 = facerec.compute_face_descriptor(img1, shape1)
            best_match = 1

            for k2, d2 in enumerate(dets2):
                # Get the landmarks/parts for the face in box d1.
                shape2 = sp(img2, d2)

                # Compute the 128D vector that describes the face in img identified by
                # shape.
                face_descriptor2 = facerec.compute_face_descriptor(img2, shape2)

                face_distance = euclidean_dist(face_descriptor1, face_descriptor2)
                if best_match > face_distance:
                    best_match = face_distance

            logger.debug("Closest euclidean distance between face pair is {}".format(best_match))

            if best_match < threshold:
                matchesfound+=1

        logger.warning("Number of face matches found: {}".format(matchesfound))
        return matchesfound

def test_class_init():
    detectionface = CamAiDetectionFaces()

def test_find_face_matches():
    detectionface = CamAiDetectionFaces()

    imgfile1 = './test1.jpg'
    imgfile2 = './test2.jpg'
    imgfile3 = './test3.jpg'

    img1 = dlib.load_rgb_image(imgfile1)
    img2 = dlib.load_rgb_image(imgfile2)
    img3 = dlib.load_rgb_image(imgfile3)

    numfaces, matches, unmatched = detectionface.find_face_matches(img1)
    logger.warning("Found these matches : {}".format(matches))

    numfaces, matches, unmatched = detectionface.find_face_matches(img2)
    logger.warning("Found these matches : {}".format(matches))

    numfaces, matches, unmatched = detectionface.find_face_matches(img=img3, scaling=1)
    logger.warning("Found these matches : {}".format(matches))

    detectionface.compare_faces(img1=img1, img2=img2)
    detectionface.compare_faces(img1=img1, img2=img3, threshold=0.3, scaling=1)
    detectionface.compare_faces(img1=img2, img2=img3, threshold=0.6, scaling=1)

if __name__ == '__main__':
    #test_class_init()
    test_find_face_matches()
