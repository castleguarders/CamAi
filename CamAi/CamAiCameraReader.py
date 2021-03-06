import os
import threading
from multiprocessing import Process
import queue
import time
import datetime
import numpy as np
import cv2 as cv
import random
import logging
import imutils

from . import CamAiMessage

logger = logging.getLogger(__name__)
#logger.setLevel(logging.WARNING)
#formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
formatter = logging.Formatter('%(asctime)s:%(name)s:%(funcName)s:%(lineno)d:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
#formatter = logging.Formatter('%(asctime)s:%(message)s')

#file_handler = logging.FileHandler('CamAiCameraWriter.errorlog')
#file_handler.setFormatter(formatter)
#file_handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
#stream_handler.setLevel(logging.DEBUG)
stream_handler.setLevel(logging.ERROR)
#stream_handler.setLevel(logging.WARNING)

#logger.addHandler(file_handler)
logger.addHandler(stream_handler)

DEBUG = False

# Give main thread time to be ready and waiting for inference
# so frames don't get lost in the beginning
Camera_Startup_Wait = 20
priming_reconnect_attempts = 3
priming_read_attempts = 5
priming_connect_wait = 4
priming_read_wait = 1

Skip_Detection = False
# Only used when Detection is being skipped
Enable_Disk_Recording_During_Skip = True
Record_Batch_Size = 1 # Batch size used in recording only mode

class CamAiCameraReader(object):

    def __init__(
                self,
                config,
                ):

        self.config = config

    def start(self):
        cname = "Reader: " + self.config.name
        if self.config.multiprocessing_reader is True:
            self.reader = Process(
                target=self._read_camera, args=([]), name=cname)
        else:
            self.reader = threading.Thread(
                target=self._read_camera, args=([]), name=cname)
            self.reader.do_read= True

        logger.warning(f"{cname} : Starting")
        self.reader.start()

    def stop(self):
        cname = "Reader: " + self.config.name
        if self.config.multiprocessing_reader is False:
            self.reader.do_read = False
        logger.error(f"{cname} : Stopping")

    def join(self, waittime=30):
        cname = "Reader: " + self.config.name
        self.reader.join(waittime)
        logger.warning(f"{cname} : Joining")

    def set_video_codec(self):
        cname = "Reader: " + self.config.name
        # @todo Trying to control codec through env var while using multiple threads
        # can result in a race condition, use multiprocessing observers
        # instead to guarantee this instead
        if self.config.ffmpeg_options != "":
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = self.config.ffmpeg_options
            logger.debug(f"{cname}: Set ffmpeg options to {self.config.ffmpeg_options}")
        else:
            logger.debug(f"{cname} No ffmpeg options set, deleting env variable")
            try:
                del os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']
            except KeyError:
                pass
        try:
            logger.warning(f"{cname}: Using codec {os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']}")
        except BaseException:
            logger.warning(f"{cname}: No codec option set, using OS default")

    # @profile
    def _read_camera(self):

        camera_source = self.config.url
        name = "Reader: " + self.config.name
        reader_queue = self.config.reader_queue
        writer_queue = self.config.writer_queue
        try:
            watchdogenable = self.config.watchdogenable
            watchdogtimer = self.config.watchdogtimer
        except AttributeError:
            watchdogtimer = 0

        logger.debug(f"{name}: Reader is starting")

        # Set the reader process to high priority to reduce frame processing latency,
        from .CamAiUtils import increase_my_priority
        increase_my_priority(20)

        priming_success = False
        reader = threading.currentThread()

        # Wait a bit for the other workers to be ready to handle the work,
        # otherwise we might hang at decoding, or lose image mb's or slices
        time.sleep(Camera_Startup_Wait)
        Burner_Frames = 100
        Queue_Put_Timeout = 0
        for connect_attempt in range(priming_reconnect_attempts):
            self.set_video_codec()
            video_capture = cv.VideoCapture(camera_source)
            video_capture.set(cv.CAP_PROP_BUFFERSIZE, self.config.readbuffer)

            for read_attempt in range(priming_read_attempts):
                if (video_capture.isOpened() == False):
                    logger.error(f"{name}: Video Capture is not in opened state ")
                # Do a priming read of an image from the camera to determine the
                # image dimensions
                ret, frame = video_capture.read()
                # if (ret==True) and (frame != None):
                # Need a more reliable decode engine .. return code is false
                # negatives a lot of times
                try:
                    if ret is True:
                        if np.any(frame):
                            priming_success = True
                            last_frame_read_time = time.time()
                            break  # Success
                        else:
                            # Burn a few frames to see if we get past cv
                            # priming issue
                            for i in range(Burner_Frames):
                                ret, frame = video_capture.read()
                                if np.any(frame):
                                    break
                                else:
                                    logger.warning(f"{name} No ndarray yet, frame type is : {type(frame)}")
                    else:
                        logger.warning(f"{name}: Failed getting a priming frame, will retry in {priming_read_wait} seconds, read attempt {read_attempt} ")
                        time.sleep(priming_read_wait)
                    # Frame is none in this case
                except AttributeError:
                    logger.warning(f"{name}: Read from source came back with a null")
                    time.sleep(priming_read_wait)

                # Check if parent requests us to stop, need to check for consistent exit handling later
                # TODO Make this MultiProcess aware
                # TODO: In progress exit notifications only through queues
                # if (reader.do_read == False):
                #    logger.warning("Stopping thread {}".format(reader.name))
                #    video_capture.release()
                #    return

            if priming_success is True:
                break
            else:
                video_capture.release()
                time.sleep(priming_connect_wait)
                logger.warning(f"{name}: Reconnecting to camera, before retrying, connect attempt: {connect_attempt} ")
                self.set_video_codec()
                video_capture = cv.VideoCapture(camera_source)
                video_capture.set(cv.CAP_PROP_BUFFERSIZE, self.config.readbuffer)
                if (video_capture.isOpened() == False):
                    logger.error(f"{name}: Video Capture is not in opened state ")

        if priming_success is False:
            logger.warning(f"{name}: Failed to prime camera, ending thread")
            video_capture.release()
            return

        num_frames_read = 0
        frame_process_time = 0
        max_latency = 0
        min_latency = 0

        do_read = True

        direct_write_cache = []
        direct_bufcount = 0
        read_failure_count = 0
        read_success_count = 0

        from .CamAiConfig import CamAiCameraMode

        while do_read is True:
            if self.config.multiprocessing_reader is False:
                # TODO: In progress exit notifications only through queues, only
                # reader still depends on do_ variable to exit
                do_read = reader.do_read
                if do_read is False:
                    logger.error(f"{name}: do_reader is set to False")
                    break

            try:
                rdstart = time.perf_counter()
                ret, frame = video_capture.read()
                if ret is True:
                    # This is happening even after priming when reading video
                    # files
                    if np.any(frame) is False:
                        logger.warning(f"{name}: np.any() returns False for frame of type {type(frame)}")
                        next

                    num_frames_read += 1
                    last_frame_read_time = time.time()
                    read_success_count  += 1

                    # Send directly to writer if detection is not enabled
                    if (self.config.mode == CamAiCameraMode.record_only.name):
                        if self.config.rotation != 0:
                            # 2x faster than cv2 for 90/270 degrees
                            frame = imutils.rotate_bound(frame, self.config.rotation)
                        direct_write_cache.append(frame)
                        direct_bufcount += 1
                        if direct_bufcount >= Record_Batch_Size:
                            message = CamAiMessage.CamAiImageList(direct_write_cache)
                            try:
                                # logger.debug("{}: Added to writer queue".format(name))
                                writer_queue.put(message)
                            except queue.Full:
                                logger.warning(f"{name}: Writer queue is full")
                                continue
                            # Can't clear the old one as it's in the queue, get a new one
                            # Old one should be auto deref'd after writer processes it
                            direct_bufcount = 0
                            direct_write_cache = []
                    # Send to observer thread for detection
                    else:
                        try:
                            message = CamAiMessage.CamAiImage(frame)
                            #reader_queue.put(message, timeout=Queue_Put_Timeout)
                            reader_queue.put(message)
                        except queue.Full:
                            logger.warning(f"{name}: Reader queue is full")
                            continue

                    latency = (time.perf_counter() - rdstart)
                    if min_latency == 0 or latency < min_latency:
                        min_latency = latency
                    if max_latency == 0 or latency > max_latency:
                        max_latency = latency
                    frame_process_time += latency
                else:
                    read_failure_count  += 1
                    if (read_failure_count%100 == 0):
                        logger.error("{}: failure count {} of {}".format(name, read_failure_count, read_success_count))

                # Check for hung video captures
                if (time.time() - last_frame_read_time) > watchdogtimer:
                    if watchdogenable is True:
                        logger.warning(f"{name}: Video capture on seems to have hung, reconnecting")
                        video_capture.release()
                        time.sleep(priming_connect_wait)
                        self.set_video_codec()
                        video_capture = cv.VideoCapture(camera_source)
                        video_capture.set(cv.CAP_PROP_BUFFERSIZE, self.config.readbuffer)
                    else:
                        logger.warning(f"{name}: Video capture on seems to have hung, and watchdog is disabled, exiting reader")
                        break

            # This shouldn't be necessary unless we put this in a different process
            except KeyboardInterrupt:
                logger.warning(f"{name}: Got a keyboard interrupt, is exiting")
                break

        # In case observer is waiting on us when record_only is enabled
        # Shouldn't be the case as observer gets quit message via oob and is
        # notifying us in the first place
        #quitmessage = CamAiMessage.CamAiQuitMsg()
        #reader_queue.put(quitmessage)
        #writer_queue.put(quitmessage)
        #logger.warning(
        #    "{}: Sent quit message via reader queue to observer or writer before exiting ".format(name))

        if video_capture is not None:
            logger.error(f"{name}: Releasing video_capture")
            video_capture.release()
            logger.error(f"{name}: Released video_capture")
        if self.config.multiprocessing_reader is True:
            reader_queue.close()

        if DEBUG is True:
            self.dump_stats(name, num_frames_read, frame_process_time, max_latency, min_latency)

        logger.error(f"{name}: Returning from reader {reader.name}")
        return

    def dump_stats(self, name, num_frames_read, frame_process_time, max_latency, min_latency):
        logger.warning(f"{name}: =============================================================================")
        logger.warning(f"{name}: Camera Reader Statistics")
        if num_frames_read > 0:
            logger.warning("{}: Average per frame processing latency is {:.2f} ms, frames read: {}" .format(
                    name, 1000*(frame_process_time/num_frames_read), num_frames_read))
        logger.warning(
            "{}: Max per frame processing latency is {:.2f} ms".format(
                name, 1000*max_latency))
        logger.warning(
            "{}: Min per frame processing latency is {:.2f} ms".format(
                name, 1000*min_latency))
        logger.warning(f"{name}: =============================================================================")

if __name__ == '__main__':
    pass
