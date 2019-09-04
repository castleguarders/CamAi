# Custom code to visualize real time video using OpenCV
import os
import sys
import time
import threading
from multiprocessing import Process, Queue
import queue
import logging
import signal
# for debug
import objgraph

from . import CamAiMessage


__version__ = '1.0'

# Logging Setup
logger = logging.getLogger(__name__)
#formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
formatter = logging.Formatter('%(asctime)s:%(name)s:%(funcName)s:%(lineno)d:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

#file_handler = logging.FileHandler('CamAi.errorlog')
#file_handler.setFormatter(formatter)
#file_handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
#stream_handler.setLevel(logging.WARNING)
stream_handler.setLevel(logging.ERROR)

#logger.addHandler(file_handler)
logger.addHandler(stream_handler)
# logger.setLevel(logging.WARN)
# logger.setLevel(logging.DEBUG)


DEBUG = False

cam_ai_configfile = None

Quit_Wait_Time = 10

class CamAi(object):

    def __init__(
                self,
                config=None,
                config_file=None
                ):

        if config is not None:
            self.config = config
        elif config_file is not None:
            self.config = self.get_config(config_file)
        else :
            logger.ERROR("No configuration object or file passed, init failed")
            raise ValueError

        self.camera_detect_queue = None

    def get_config(self, config_file):
        from .CamAiConfig import CamAiConfig
        camaiconfig = CamAiConfig(config_file)
        return camaiconfig

    def create_camera_queues(self, cameraconfig):
        manager_options = self.config.get_manager_options()
        maxqueuesize = manager_options['defaultmaxqueue']

        # For CamAiCameras to talk to detectors
        if manager_options['multiprocessing_detector'] is True:
            if (manager_options['singledetectorqueue'] is True):
                if self.camera_detect_queue is None:
                    self.camera_detect_queue = Queue()
                camera_detect_queue = self.camera_detect_queue
            else:
                camera_detect_queue = Queue()

            camera_response_queue = Queue()
        else:
            if (manager_options['singledetectorqueue'] is True):
                if self.camera_detect_queue is None:
                    self.camera_detect_queue = queue.Queue(maxqueuesize)
                camera_detect_queue = self.camera_detect_queue
            else:
                camera_detect_queue = queue.Queue(maxqueuesize)

            camera_response_queue = queue.Queue(maxqueuesize)

        # For CamAiCameras to issue notifications
        if manager_options['multiprocessing_notifier'] is True:
            camera_notification_queue = Queue()
        else:
            camera_notification_queue = queue.Queue(maxqueuesize)

        # For CamAi to communicate with cameras
        if manager_options['multiprocessing_observer'] is True:
            camera_oob_queue = Queue()
        else:
            camera_oob_queue = queue.Queue(maxqueuesize)

        # For CamAiCamera to talk to it's reader
        if manager_options['multiprocessing_reader'] is True:
            camera_reader_queue = Queue()
        else:
            # Try/Except block TODO:
            maxreadqueue = 64
            camera_reader_queue = queue.Queue(maxreadqueue)

        # For CamAiCamera to talk to it's writer
        if manager_options['multiprocessing_writer'] is True:
            camera_writer_queue = Queue()
        else:
            # Try/Except block TODO:
            maxreadqueue = 64
            camera_writer_queue = queue.Queue(maxreadqueue)

        camera_queues = {
                        'detect_queue': camera_detect_queue,
                        'response_queue': camera_response_queue,
                        'notification_queue': camera_notification_queue,
                        'oob_queue': camera_oob_queue,
                        'reader_queue': camera_reader_queue,
                        'writer_queue': camera_writer_queue
                }

        return camera_queues

# @profile


def start_cameras(config_file):
    camera_names = []
    camera_detect_queues = []
    camera_response_queues = []
    camera_oob_queues = []
    camera_notification_queues = []  # Overkill to have one per cam, rethink
    aicameras = {}
    children = []

    if config_file is None:
        print ("Please specify a configuration file write to")
        return

    if os.path.isfile(config_file) is False:
        print("Configuration file: {} doesn't exist, exiting.".format(config_file))
        return

    from .CamAiNotification import CamAiNotification

    self = CamAi(config_file=config_file)
    if self.config is None:
        logger.error("Initialization of CamAi configuration failed")
        logger.error("Incorrect or missing configuration file specified?")
        return

    manager_options = self.config.get_manager_options()

    # Just above others, but readers should get most scheduling
    from .CamAiUtils import increase_my_priority
    increase_my_priority(5)

    camera_index = 0
    cameraconfigs = self.config.get_cameras()
    for cameraconfig in cameraconfigs:
        # These should go into the loop as well, but URL is mandatory
        cameraname =  cameraconfig['name']

        camera_names.append(cameraname)

        camera_handle = camera_index
        # Need to abstract this list and not rely on indexes, have a cammgr and
        # camera class instead of this mess
        camera_index += 1

        camera_queues = self.create_camera_queues(cameraconfig)

        # TODO, not all queues are initialized till the end of this loop
        # AiCameras need detection config for image resize pipelining
        # Need a cleaner way to do this and get rid of the catch-22 dependency
        from .CamAiDetection import CamAiDetection
        detection = CamAiDetection(
            name=cameraname, detect_queues=camera_detect_queues,
            response_queues=camera_response_queues,
            pipeline_image_resize=manager_options['pipelineresize'],
            singledetectorqueue=manager_options['singledetectorqueue'],
            multiprocessing=manager_options['multiprocessing_detector'])

        from .CamAiCamera import CamAiCamera
        aicamera = CamAiCamera.from_dict(
            camera_handle=camera_handle, cameraconfig=cameraconfig,
            managerconfig=manager_options, camera_queues=camera_queues,
            detection=detection)

        # TODO: Manager Should just use aicamera objects instead of instead of
        # discrete arrays for each queue type
        camera_detect_queues.append(camera_queues['detect_queue'])
        camera_response_queues.append(camera_queues['response_queue'])
        camera_oob_queues.append(camera_queues['oob_queue'])
        camera_notification_queues.append(camera_queues['notification_queue'])

        aicameras[camera_handle] = aicamera

        # # TODO: Redundant with aicameras, need to refactor
        children.append(aicamera)
        logger.warning("{}: Start camera".format(cameraname))
        aicamera.start()

    name = "Camera Manager: "

    # Start the notification process/thread, face detection needs another
    # tensorflow instance, so a process is required for now,
    # Doing this before starting detectors should ensure GPU memory gets
    # allocated to facedetection before detectors greedy allocation for maskrcnn
    # takes up everything available
    # TODO: need to evaluate how to consolidate this into detector.
    # TODO: Looks like the intermittent notifier process hanging at start
    # is tied to a combination of multithreading/processing/logging in the same
    # process at the same time Reducing logging around notification process creation
    # 'could' reduce the occurence
    notification = CamAiNotification(self.config, camera_notification_queues)
    #notification = CamAiNotification.CamAiNotification(self.config, camera_notification_queues)
    notification.start()

    # This stuff should move to CamAiDetection class's start method
    detectors = []
    if manager_options['multiprocessing_detector'] is False:
        num_detectors = manager_options['numdetectors']
    else:
        num_detectors = 1

    # for num in range(Number_Of_Detectors):
    # TODO: Yikes, we are using threads directly instead of detector classes
    # Need to finish this last bit of conversion to classes
    from .CamAiDetection import object_detector_server
    for num in range(num_detectors):
        if (manager_options['multiprocessing_detector'] is False):
            detector = threading.Thread(
                target=object_detector_server,
                # detector =
                # ProfiledThread(target=object_detector_server,
                # \
                args=(detection, camera_names),
                name=("detector" + "_" + str(num)))
            detector.do_detect = True
            detectors.append(detector)
            logger.warning("{}: Starting detector thread number {}".format(name, num))
            detector.start()

        else:
            detector = Process(
                target=object_detector_server,
                # detector =
                # ProfiledProcess(target=object_detector_server,
                # \
                args=(detection, camera_names),
                name=("detector" + "_" + str(num)))
            detectors.append(detector)
            logger.warning("{}: Starting detector process number {}".format(name, num))
            detector.start()

    logger.debug("{}: Installing Signal Handlers".format(name))
    install_sighandler()

    waitcount = 0

    global signal_received_SIGHUP
    global signal_received_SIGQUIT
    global signal_received_SIGTERM

    signal_received_SIGHUP = False
    signal_received_SIGQUIT = False
    signal_received_SIGTERM = False

    while True:
        try:
            time.sleep(5)

            if DEBUG is True:
                # Debug memory leaks, print every 5 minutes
                waitcount += 1
                if waitcount % 120 == 0:
                    objgraph.show_most_common_types(limit=20)

            # Handle signals here
            if signal_received_SIGHUP is True:
                # TODO: Should reread config files, tell writer to rotate logs
                # etc.
                logger.warning("{}: TODO: SIGHUP is not yet handled".format(name))
                signal_received_SIGHUP = False
                pass
            if signal_received_SIGQUIT is True:
                logger.warning("{}: SIGQUIT received, exiting".format(name))
                break
            if signal_received_SIGTERM is True:
                logger.warning("{}: SIGTERM received, exiting".format(name))
                break

        except KeyboardInterrupt:
            break

    # Start Cleaning up all child threads/processes here
    quitmessage = CamAiMessage.CamAiQuitMsg()

    # Let Observers know via queues
    for index, aicamera in enumerate(children):
        logger.warning(
            "{}: sending quit message on oob queue for {}".format(
                name, aicamera.name))
        camera_oob_queues[index].put(quitmessage, False)
        aicamera.stop()

        # Tell threads/processes to not wait for any more queue items
        #camera_response_queues[index].put(quitmessage, False)
        #aicamera.join(10)

    # Let Detectors know via queues
    for camera_detect_queue in camera_detect_queues:
        logger.warning("{}: Sending Quit to detect queues".format(name))
        camera_detect_queue.put(quitmessage)
        # Detectors should in turn let observers know via response queue if
        # they are in the middle of processing, or should they even do this?

    # Wait on observers
    for index, aicamera in enumerate(children):
        logger.warning("{}: Waiting on Camera {} to return".format(name, aicamera.name))
        aicamera.stop()
        aicamera.join(10)

    for detector in detectors:
        detector.do_detect = False # TODO: get rid of this when fully moved to detection class usage
        #detector.stop()
        detector.join(10)

    # Notifier, observers should already have asked corresponding notifiers
    # via respective notification_queues
    notification.stop()
    notification.join(10)

    if DEBUG is True:
        objgraph.show_most_common_types(limit=30)

    os.sync()
    #cv.destroyAllWindows()

    logger.warning("{}: Bye, cleaning up in {} seconds".format(name, Quit_Wait_Time))
    for t in range(Quit_Wait_Time):
        logger.warning("{}: {} seconds left".format(name, (Quit_Wait_Time-t)))
        time.sleep(1)
        os.sync() # Flush whatever buffers we can till the end


def install_sighandler():
    # register the signals to be caught
    signal.signal(signal.SIGHUP, handle_sighup)
    signal.signal(signal.SIGQUIT, handle_sigquit)
    signal.signal(signal.SIGTERM, handle_sigterm)


def handle_sighup(signalNumber, frame):
    global signal_received_SIGHUP
    signal_received_SIGHUP = True
    logger.warning("received signal HUP: {}".format(signalNumber))


def handle_sigquit(signalNumber, frame):
    global signal_received_SIGQUIT
    signal_received_SIGQUIT = True
    logger.warning("received signal QUIT: {}".format(signalNumber))


def handle_sigterm(signalNumber, frame):
    global signal_received_SIGTERM
    signal_received_SIGTERM = True
    logger.warning("received signal TERM: {}".format(signalNumber))


def handle_signal(signalNumber, frame):
    logger.warning("received signal: {}".format(signalNumber))


def empty():
    print("Empty EOF")
