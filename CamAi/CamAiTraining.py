import numpy as np
import queue
import threading
from multiprocessing import Process
import logging
import datetime
import cv2 as cv

#import CamAiUtils
#import CamAiMessage
#import CamAiDetection

logger = logging.getLogger(__name__)
#logger.setLevel(logging.WARN)
#formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
formatter = logging.Formatter('%(asctime)s:%(message)s')

#file_handler = logging.FileHandler('CamAiCameraWriter.errorlog')
#file_handler.setFormatter(formatter)
#file_handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)

#logger.addHandler(file_handler)
logger.addHandler(stream_handler)

DEBUG = False


class CamAiTraining(object):

    def __init__(
                self,
                config,
                trainer_queue
                ):

        self.config = config
        self.trainer_queue = trainer_queue

    @property
    def basedir(self):
        return self.config.get_basedir()

    def run_as_process(self):
        manager_options = self.config.get_manager_options()

        if (manager_options['multiprocessing_trainer'] is True):
            return True
        else:
            return False

    def start(self):
        if self.run_as_process():
            self.trainer = Process(target=self._do_training,
                                   args=([]),
                                   name="trainer")

        else:
            self.trainer = threading.Thread(target=self._do_training,
                                            args=([]),
                                            name="trainer")
            self.trainer.do_training = True
        logger.warn("Starting trainer")
        self.trainer.start()

    def stop(self):
        self.trainer.do_training = False
        logger.warn("Stopping trainer")

    def join(self, waittime=10):
        self.trainer.join(waittime)
        logger.warn("Join trainer")

    def _do_training(self):
        training_wait = 2

        trainer = threading.currentThread()
        # Parent cannot set this if invoked as a process
        if (self.run_as_process() is True):
            do_training = True

        # while trainer.do_training is True :
        while do_training is True:
            try:
                if (self.run_as_process() is False):
                    do_training = trainer.do_training

                try:
                    message = self.trainer_queue.get(True, training_wait)
                    if message.msgtype == CamAiMessage.CamAiMsgType.notification:
                        # microseconds are too noisy to keep around for
                        # notifications
                        message.msgdata['timestamp'] = message.msgdata['timestamp'] - \
                                datetime.timedelta(microseconds=message.msgdata['timestamp']
                                                   .microsecond)

                        image = message.msgdata['image']
                        matches = message.msgdata['objects detected']
                        confidence = False
                        (h, w) = image.shape[:2]
                        for match in matches:
                            logger.warn(
                                "Match class: {}, score: {}, roi: {}".format(
                                    match['class'], match['score'], match['roi']))
                            y1, x1, y2, x2 = match['roi']
                            color = tuple(255 * np.random.rand(3))
                            image = cv.rectangle(
                                image, (x1, y1), (x2, y2), color, 2)
                            # Make the crop square so we can feed more of the pertinent image than
                            # zero pads, We also expand the area by 21% to account
                            # for bounding boxes being smaller than actual objects. This happens
                            # quite often when objects are partially occluded
                            cy1 = int(y1*.9)
                            cx1 = int(x1*.9)
                            cy2 = int(y2*1.1)
                            cx2 = int(x2*1.1)
                            if (cy2-cy1) > (cx2-cx1):
                                cx2 = min(cx1+(cy2-cy1), w)
                            elif (cx2-cx1) > (cy2-cy1):
                                cy2 = min(cy1+(cx2-cx1), h)

                            objectcrop = image[cy1:cy2, cx1:cx2]

                            logger.warn(
                                "Starting to do second level object detection")
                            # Ensemble detection from hell, wait for keras 2.2.5 for resnext
                            # resnext_results = CamAiDetection.get_objects_ResnetNeXt101(objectcrop)
                            # incepresv2_results = CamAiDetection.get_objects_IncepResV2 (objectcrop)
                            # resnet50_results = CamAiDetection.get_objects_Resnet50 (objectcrop)
                            # logger.warn("Resnext detected: {}".format(resnext_results))
                            # logger.warn("Incepresv2 detected: {}".format(incepresv2_results))
                            # logger.warn("Resnet50 detected: {}".format(resnet50_results))

                            logger.warn("Starting to do face detection")
                            # Try to do face detection to weed out false
                            # positives
                            numfaces1, numeyes1, faceimage1 = CamAiDetection.get_faces(
                                objectcrop)
                            alarm_image_file = self.basedir + message.msgdata['cameraname'] \
                                + "_alarm_faces1" + str(message.msgdata['timestamp']) + ".png"
                            rc = cv.imwrite(alarm_image_file, faceimage1)

                            numfaces2, numeyes2, faceimage2 = CamAiDetection.get_faces2(
                                objectcrop)
                            alarm_image_file = self.basedir + message.msgdata['cameraname'] \
                                + "_alarm_faces2" + str(message.msgdata['timestamp']) + ".png"
                            rc = cv.imwrite(alarm_image_file, faceimage2)

                            numfaces3, numeyes2, faceimage3 = CamAiDetection.get_faces3(
                                objectcrop)
                            alarm_image_file = self.basedir + message.msgdata['cameraname'] \
                                + "_alarm_faces3" + str(message.msgdata['timestamp']) + ".png"
                            rc = cv.imwrite(alarm_image_file, faceimage3)

                            if numfaces1 > 0 or numfaces2 > 0 or numfaces3 > 0:
                                confidence = True

                        alarm_image_file = self.basedir + message.msgdata['cameraname'] \
                            + "_alarm_" + str(message.msgdata['timestamp']) + ".png"
                        rc = cv.imwrite(alarm_image_file, image)
                        # rc = cv.imwrite(alarm_image_file, message.msgdata['image'])

                    else:
                        logger.warn("Empty message for trainer to process?")
                        pass
                except queue.Empty:
                    logger.debug(
                        "Notifier queue is empty for {} ".format(
                            trainer.name))
                except AttributeError as ae:
                    logger.warn(
                        "AttributeError in trainer loop, shouldn't happen : {}\n{} ". format(
                            trainer.name, ae))
            except KeyboardInterrupt:
                logger.warn(
                    "Got a keyboard interrupt, {} is exiting".format(
                        trainer.name))
                break

        logger.warn("Returning from trainer: {}".format(trainer.name))
        return
