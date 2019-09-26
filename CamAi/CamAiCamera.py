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
import scipy

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import CamAiMessage
from . import CamAiDetection

logger = logging.getLogger(__name__)
#logger.setLevel(logging.WARNING)
logger.setLevel(logging.ERROR)
#formatter = logging.Formatter('%(asctime)s:%(message)s')
#formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
formatter = logging.Formatter('%(asctime)s:%(name)s:%(funcName)s:%(lineno)d:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

#file_handler = logging.FileHandler('CamAiCameraWriter.errorlog')
#file_handler.setFormatter(formatter)
#file_handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
#stream_handler.setLevel(logging.WARNING)
stream_handler.setLevel(logging.ERROR)

#logger.addHandler(file_handler)
logger.addHandler(stream_handler)

DEBUG = False

Q_Depth_Profiler_Interval = 300
Alarm_Cache_Size = 180 # Batch size used when events are detected

class CamAiCamera (object):

    def __init__(
                self,
                handle,
                # url,
                queues=None,
                detection=None,
                camvars = None,
                managervars = None
                ):

        self.start_time = time.time()

        # Camera Options
        self.handle = handle

        # Detection object for detection state dependant functions
        self.detection = detection

        if camvars and len(camvars) > 0 :
            for key in camvars:
                logger.debug(f"Adding camera attribute: {key}, with value: {camvars[key]}")
                setattr(self, key, camvars[key])

        if managervars and len(managervars) > 0 :
            for key in managervars:
                logger.debug(f"Adding manager attribute: {key}, with value: {managervars[key]}")
                setattr(self, key, managervars[key])

        if queues is not None:
            self.reader_queue = queues['reader_queue']
            self.writer_queue = queues['writer_queue']
            self.detect_queue = queues['detect_queue']
            self.response_queue = queues['response_queue']
            self.oob_queue = queues['oob_queue']
            self.notification_queue = queues['notification_queue']
        else:
            logger.warning(f"Camera {name}: Queues are not initialized yet")

        if self.subdir is None:
            self.mydir = os.path.join(self.basedir,'')
        else:
            self.mydir = os.path.join(self.subdir,'')

        if not os.path.exists(self.mydir):
            os.makedirs(self.mydir)

        self.odb = None

    @classmethod
    def from_dict(cls, camera_handle, cameraconfig,
                  managerconfig, camera_queues, detection):
        from .CamAiConfig import CAMVARS, MANAGERVARS
        camvars = CAMVARS
        #camvars = CamAiConfig.CAMVARS

        # Update mandatory configuration variables:defaults with user provided values if any
        for key in camvars:
            try:
                camvars[key] = cameraconfig[key]
            except KeyError:
                logger.warning(f"{camvars['name']}: key: {key} doesn't exist in the config file, going with default {camvars[key]}")

        # Add variables from configuration that are optional, TODO: Might not
        # need this, handy while experimenting new variables
        for key in cameraconfig:
            try:
                camvars[key] = cameraconfig[key]
            except KeyError:
                logger.warning(f"{camvars['name']}: key: {key} doesn't exist in defaults , going with user provided value{cameraconfig[key]}")

        managervars = MANAGERVARS
        #managervars = CamAiConfig.MANAGERVARS
        for key in managervars:
            try:
                managervars[key] = managerconfig[key]
            except KeyError:
                logger.warning(f"Manager {managervars['name']}: key: {key} doesn't exist in the config file, going with default {managervars[key]}")

        for key in managerconfig:
            try:
                managervars[key] = managerconfig[key]
            except KeyError:
                logger.warning(f"Manager {managervars['name']}: key: {key} doesn't exist in the config file, going with default {managervars[key]}")

        return CamAiCamera(
            handle=camera_handle,
            queues=camera_queues,
            detection=detection,
            camvars = camvars,
            managervars = managervars
        )

    def get_handle(self):
        return self.handle

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, angle):
        if angle >= 0 and angle <= 360:
            self._rotation = angle
        else:
            self._rotation = 0

    def start(self):
        cname = "Observer: " + self.name
        if self.multiprocessing_observer is True:
            self.observer = Process(
                target=self._observe_camera, args=([]), name=cname)
        else:
            self.observer = threading.Thread(
                target=self._observe_camera, args=([]), name=cname)
            self.observer.do_observe = True

        logger.warning(f"{cname} : Starting")
        self.start_time = time.time()
        self.observer.start()
        #self.start_video_player()

    def stop(self):
        cname = "Observer: " + self.name
        if self.multiprocessing_observer is False:
            self.observer.do_observe = False
        logger.warning(f"{cname} : Stopping")
        #self.stop_video_player()

    def join(self, waittime=10):
        cname = "Observer: " + self.name
        self.observer.join(waittime)
        logger.warning(f"{cname} : Joining")

    def start_video_player(self):
        cname = "Observer: " + self.name
        try:
            if True:
                pass
                #self.video_player = CamAiViewer.play_stream(self.url)
        except:
            logger.error(f"{cname} : vlc exception: maybe this camera does not support multiple streams")

    def stop_video_player(self):
        cname = "Observer: " + self.name
        try:
            if self.video_player:
                #logger.error(f"{cname} : starting video player with url: {self.url}")
                self.video_player.stop()
        except:
            logger.error(f"{cname} : vlc exception: issue stopping video player")

    def get_object_timeseries(self):
        name = "Observer: " + self.name
        parquet_file = os.path.join(self.mydir, self.name + '_events.parquet')
        if os.path.isfile(parquet_file):
            dbtable = pq.read_table(parquet_file)
            odb = dbtable.to_pandas()
            if odb is None:
                logger.error(f"{name} : odb is None after loading {parquet_file}")
            else:
                odb['object_name'].astype('str', copy=False)
                odb['confidence'].astype('float64', copy=False)
                odb['boundingbox_x1'].astype('int64', copy=False)
                odb['boundingbox_y1'].astype('int64', copy=False)
                odb['boundingbox_x2'].astype('int64', copy=False)
                odb['boundingbox_y2'].astype('int64', copy=False)

                logger.debug(f"{name} : odb types: {odb.dtypes}")
        else:
            # First time, so create the db dataframe
            col_names = ['detect_time',
                         'object_name',
                         'confidence',
                         'boundingbox_x1',
                         'boundingbox_y1',
                         'boundingbox_x2',
                         'boundingbox_y2'
                        ]

            odb = pd.DataFrame(columns=col_names)
            odb['detect_time'] = pd.to_datetime(odb['detect_time'])
            odb['object_name'].astype('str', copy=False)
            odb['confidence'].astype('float64', copy=False)
            odb['boundingbox_x1'].astype('int64', copy=False)
            odb['boundingbox_y1'].astype('int64', copy=False)
            odb['boundingbox_x2'].astype('int64', copy=False)
            odb['boundingbox_y2'].astype('int64', copy=False)
            odb = odb.set_index('detect_time')

        self.odb = odb
        logger.debug(f"{name} : odb entries: {self.odb}")


    def update_object_timeseries(self, detect_time, matches, write=False):
        if self.odb is None:
            self.get_object_timeseries()
        odb = self.odb
        name = "Observer: " + self.name

        if matches is not None:
            for match in matches:
                logger.debug("Observer: Match class: {match['class']}, score: {match['score']}, roi: {match['roi']}")
                # y1, x1, y2, x2 = boxes[i]
                odb.loc[detect_time] = [match['class'],
                                        match['score'],
                                        match['roi'][1],
                                        match['roi'][0],
                                        match['roi'][3],
                                        match['roi'][2]
                                       # match['roi']
                                        ]

        if write is True:
            logger.debug(f"{name} : odb is {odb}")
            parquet_file = os.path.join(self.mydir, self.name + '_events.parquet')
            if os.path.isfile(parquet_file):
                logger.debug (f"{name}:File exists, will overwrite")
            odbtable = pa.Table.from_pandas(odb)
            pq.write_table(odbtable, parquet_file)

        self.odb = odb

    # Return the ROI bounding box coordinates of an object seen at
    # between oldest 'start' time and latest time 'end'.
    # 'start' and 'end' are specified in minutes relative to 'now'
    # e.g: start = 0 and end = 0 when it was last seen.
    #      start = 30 and end = 0 when it was seen in the last 30 minutes
    #      start = 45 and end = 15 when it was seen between 45 minutes ago and
    #      15 minutes ago
    # time range should not exceed 60  minutes
    # E.g. objects  'car' , 'bicycle'
    def get_lastseen_roi(self, object_name, start=0, end=0):
        if self.odb is None:
            self.get_object_timeseries()
        odb = self.odb
        name = "Observer: " + self.name

        #logger.debug(f"Searching for object {object_name} in {odb}")

        #obj_match = odb.loc[odb['object_name'] == name].index.max()
        obj_match = odb.loc[odb['object_name'] == object_name]
        #logger.debug(f"Object match list is {obj_match}")
        if start is 0 and end is 0:
            obj_match_latest = obj_match.max()
        else:
            now = datetime.datetime.now()
            start_datetime = now - datetime.timedelta(minutes=start)
            end_datetime = now - datetime.timedelta(minutes=end)
            start_time = datetime.time(start_datetime.hour, start_datetime.minute, start_datetime.second)
            end_time = datetime.time(end_datetime.hour, end_datetime.minute, end_datetime.second)

            obj_match_range = obj_match.between_time(start_time, end_time)
            obj_match_latest = obj_match_range.max()

        logger.debug(f"Object latest match is {obj_match_latest}")

        if np.any(obj_match):
            roi = (
                obj_match_latest['boundingbox_y1'],
                obj_match_latest['boundingbox_x1'],
                obj_match_latest['boundingbox_y2'],
                obj_match_latest['boundingbox_x2']
            )
        else:
            roi = None

        logger.debug(f"ROI being returned is {roi} ")

        return roi

    # Return if there is an ROI match better than input threshold
    # for the object passed in.
    # between oldest 'start' time and latest time 'end'.
    # 'start' and 'end' are specified in minutes relative to 'now'
    # If continuous is set to True, then the object has to be present at least
    # once each 60 seconds within the given range
    def get_roi_match_in_timerange(self, object_name, roi, start=0, end=0, match_threshold=80, continuous=True):
        Continuous_Sampling_Interval = 2 # in minutes
        if self.odb is None:
            self.get_object_timeseries()
        odb = self.odb
        name = "Observer: " + self.name

        now = datetime.datetime.now()
        start_datetime = now - datetime.timedelta(minutes=start)
        end_datetime = now - datetime.timedelta(minutes=end)
        # These are necessary for pandas dataframe date ranges
        start_time = datetime.time(start_datetime.hour, start_datetime.minute)
        end_time = datetime.time(end_datetime.hour, end_datetime.minute)

        threshold = match_threshold/100
        obj_match_name = odb.loc[odb['object_name'] == object_name]
        #logger.warning(f"Object match list is {obj_match}")

        if continuous:
            testing_intervals = int((start - end)/Continuous_Sampling_Interval)
            sampling_interval = Continuous_Sampling_Interval
        else:
            testing_intervals = 1
            sampling_interval = start - end

        logger.warning(f"Testing Intervals: {testing_intervals}, from: {start_time}, to: {end_time}")

        for interval in range(testing_intervals):
            found_in_interval = False
            interval_end_datetime = start_datetime + datetime.timedelta(minutes=(interval+1)*sampling_interval)
            interval_end_time = datetime.time(interval_end_datetime.hour, interval_end_datetime.minute)

            #logger.warning (f"Checking interval {interval}, from: {start_time}, to: {interval_end_time}")

            obj_match_range = obj_match_name.between_time(start_time, interval_end_time)

            # Avoid false absence if no entries were logged in that time range
            # Hard to determine if absence is because we didn't log, or if it
            # wasn't detected, so commenting out for now
            #if obj_match_range.empty:
            #    logger.warning(f"No entries were logged between {start_time} and {interval_end_time} time")
            #    next

            for index, obj_match in obj_match_range.iterrows():
                testroi = (
                    obj_match['boundingbox_y1'],
                    obj_match['boundingbox_x1'],
                    obj_match['boundingbox_y2'],
                    obj_match['boundingbox_x2']
                )
                iou = CamAiDetection.get_roi_iou(roi, testroi)
                logger.debug(f"Testing: {object_name}, interval: {interval}, from: {start_time}, to: {interval_end_time}, iou is : {iou}")
                if iou >= threshold:
                    found_in_interval= True
                    # Not point searching more after we foud at least one match
                    # in the interval period
                    break

            if not found_in_interval:
                logger.warning(f"{object_name} wasn't present between {start_time} and {interval_end_time} time")
                break

            # Update start_time for next iteration
            start_time = interval_end_time

        return found_in_interval


    # @profile
    def _observe_camera(self):

        detection = self.detection
        reader_queue = self.reader_queue
        writer_queue = self.writer_queue
        name = "Observer: " + self.name
        rotation = self.rotation
        camera_source = self.url
        camera_handle = self.handle
        camera_detect_queue = self.detect_queue
        camera_response_queue = self.response_queue
        camera_oob_queue = self.oob_queue
        camera_notification_queue = self.notification_queue
        objects_of_interest = self.objects_of_interest

        logger.warning(f"{name}: Adding source name/URL/Handle: {camera_source} {camera_handle}")

        from .CamAiConfig import CamAiCameraMode

        from .CamAiCameraWriter import CamAiCameraWriter
        writer = CamAiCameraWriter(self)
        #writer = CamAiCameraWriter.CamAiCameraWriter(self)
        writer.start()

        from .CamAiCameraReader import CamAiCameraReader
        reader = CamAiCameraReader(self)
        #reader = CamAiCameraReader.CamAiCameraReader(self)
        reader.start()

        if DEBUG is True:
            detect_qsize_monitor = []
            response_qsize_monitor = []
            read_qsize_monitor = []
            write_qsize_monitor = []
            qprofiler_last_update_time = time.time()

        skipped_frames_cache = []

        do_observe = True  # Process get keyboard exits, cannot be told directly by parent

        logger.warning (f"{name}: Starting to operate Camera in Mode: {self.mode}")

        # Process the frames that were read
        while do_observe is True:
            if (self.multiprocessing_observer is False):
                # TODO: In progress exit notifications only through queues
                # do_observe = observer.do_observe
                pass

            if DEBUG is True:
                # Track q depths for performance profiling
                if (time.time() - qprofiler_last_update_time) > Q_Depth_Profiler_Interval:
                    detect_qsize_monitor.append(camera_detect_queue.qsize())
                    response_qsize_monitor.append(camera_response_queue.qsize())
                    read_qsize_monitor.append(reader_queue.qsize())
                    write_qsize_monitor.append(writer_queue.qsize())
                    qprofiler_last_update_time = time.time()
            try:
                # Handle Detect and Record Timelapse Mode
                if (self.mode == CamAiCameraMode.detect_and_record_timelapse.name or
                    self.mode == CamAiCameraMode.detect_and_record_everything.name or
                    self.mode == CamAiCameraMode.detect_only.name):

                    images, should_quit = self.get_images_from_camera(1)
                    if should_quit is True:
                        do_observe = False
                        break
                    frame = images[0] # TODO: Make this batching


                    molded_image, scale, padding = self.detection.resize_image(frame)
                    detect_images = [molded_image]
                    results, should_quit = self.do_detection([frame])
                    if should_quit is True:
                        do_observe = False
                        break
                    if results is None:
                        logger.warning (f"{name}: No results from detection probably shutting down")
                        continue
                    r = results[0]
                    alarm = False

                    # Are there any ROIs to process?  We might want to use
                    # these ROIs to do face detection or ptz etc
                    # TODO: Refactor this into a separate alarm function
                    if np.shape(r['rois'])[0] > 0:
                        # There are detections, should we alarm?
                        #alarm, best_match_index, matched_records = CamAiDetection.get_matches(r)
                        matched_objects_dict = CamAiDetection.get_watched_matches(r, objects_of_interest)
                        detect_time = datetime.datetime.now()
                        has_startup_wait_passed = int((time.time() - self.start_time)/60) # convert to minutes

                        for object_index in matched_objects_dict:
                            # TODO: Short term hack while refactoring is underway
                            # This needs to be functioned away, or ideally
                            # classed and extensible somehow, compound cases
                            # are 'interesting' i.e. people and cars or people
                            # on vehicles, people in cars
                            matched_records = matched_objects_dict[object_index]['matched_records']

                            if object_index is CamAiDetection.Person_Index:
                                for matched_record in matched_records:
                                    logger.debug(f"matched_record is {matched_record}")
                                    matchroi = matched_record['roi']
                                    was_present = self.get_roi_match_in_timerange(
                                        object_name=CamAiDetection.Person_Classname,
                                        roi=matchroi,
                                        start=objects_of_interest[CamAiDetection.Person_Classname]['instance_watch_timerange_start'], #  Watch_Timerange_Start,
                                        end= objects_of_interest[CamAiDetection.Person_Classname]['instance_watch_timerange_end'], # Watch_Timerange_End,
                                        match_threshold=objects_of_interest[CamAiDetection.Person_Classname]['instance_match_threshold'], # Person_ROI_Match_Threshold
                                        continuous=objects_of_interest[CamAiDetection.Person_Classname]['instance_watch_continuous_mode']
                                    )
                                    if was_present:
                                        logger.warning(f"This {CamAiDetection.coco_class_names[object_index]} was present already")
                                    else:
                                        if has_startup_wait_passed >  objects_of_interest[CamAiDetection.Person_Classname]['notify_startup_wait']:
                                            alarm = matched_objects_dict[object_index]['found']
                                            logger.warning(f"This {CamAiDetection.coco_class_names[object_index]} is new")
                                        else:
                                            logger.debug(f"Skipping notification of {CamAiDetection.coco_class_names[object_index]} as notify startup wait time hasn't passed ")

                            elif object_index in CamAiDetection.Vehicle_Indexes:
                                for matched_record in matched_records:
                                    logger.debug(f"matched_record is {matched_record}")
                                    matchroi = matched_record['roi']
                                    was_present = self.get_roi_match_in_timerange(
                                        object_name=CamAiDetection.Car_Classname,
                                        roi=matchroi,
                                        start=objects_of_interest[CamAiDetection.Car_Classname]['instance_watch_timerange_start'], #  Watch_Timerange_Start,
                                        end= objects_of_interest[CamAiDetection.Car_Classname]['instance_watch_timerange_end'], # Watch_Timerange_End,
                                        match_threshold=objects_of_interest[CamAiDetection.Car_Classname]['instance_match_threshold'], # Person_ROI_Match_Threshold
                                        continuous=objects_of_interest[CamAiDetection.Car_Classname]['instance_watch_continuous_mode']
                                    )
                                    if was_present:
                                        logger.warning(f"This {CamAiDetection.coco_class_names[object_index]} was present already")
                                    else:
                                        if has_startup_wait_passed >  objects_of_interest[CamAiDetection.Car_Classname]['notify_startup_wait']:
                                            alarm = matched_objects_dict[object_index]['found']
                                            logger.warning(f"This {CamAiDetection.coco_class_names[object_index]} is new")
                                        else:
                                            logger.debug(f"Skipping notification of {CamAiDetection.coco_class_names[object_index]} as notify startup wait time hasn't passed ")


                            # Log to object_timeseries for prediction analysis
                            if self.log_object_timeseries is True:
                                self.update_object_timeseries(detect_time, matched_records)

                    # TODO This should be conditional to resize pipelining also
                    # Mask zooming is adding more overhead/latency than
                    # expected
                    if self.annotation is True:
                        frame = CamAiDetection.annotate_objects_in_frame(
                            frame, r['rois'], r['masks'], r['class_ids'], r['scores'])

                    if (alarm is False):
                        skipped_frames_cache = []

                        if self.mode != CamAiCameraMode.detect_only.name:
                            # Send the current image to writer queue
                            message = CamAiMessage.CamAiImage(frame)
                            writer_queue.put(message)

                        skipped_frames_cache, should_quit = self.get_images_from_camera(self.detectionrate)
                        if should_quit is True:
                            do_observe = False
                            break

                        if self.mode == CamAiCameraMode.detect_and_record_everything.name:
                            batchmessage = CamAiMessage.CamAiImageList(skipped_frames_cache)
                            writer_queue.put(batchmessage)

                    # Alarm is True
                    else:
                        logger.warning(f"{name}: Logging alarm frames on camera")

                        if self.mode != CamAiCameraMode.detect_only.name:
                            # This catches the case if the first frame itself is alarming
                            if len(skipped_frames_cache) > 0:
                                batchmessage = CamAiMessage.CamAiImageList(skipped_frames_cache)
                                try:
                                    writer_queue.put(batchmessage)
                                except queue.Full:
                                    logger.warning(f"{name}: Could not send skipped frame cache")
                                    pass

                        # Now send the current frame to writer
                        if self.mode != CamAiCameraMode.detect_only.name:
                            try:
                                message = CamAiMessage.CamAiImage(frame)
                                writer_queue.put(message)
                            except queue.Full:
                                logger.warning(f"{name}: Could not send frame to writer")

                        skipped_frames_cache = []

                        # Don't wait, get the next set of tracked frames as an object we are interested in
                        # has been detected skip object detection and save
                        starttime = time.time()
                        endtime = starttime + self.detecttrackseconds
                        notified_once = False
                        while (time.time() < endtime):
                            logger.warning(f"{name}:Skipping draining of camera as alarm is detected")
                            alarm_cache = []
                            alarm_cache, should_quit = self.get_images_from_camera(Alarm_Cache_Size)
                            if should_quit is True:
                                do_observe = False
                                break

                            if self.mode != CamAiCameraMode.detect_only.name:
                                logger.debug(f"{name}: Batched {Alarm_Cache_Size} alarm frames to writer")
                                batchmessage = CamAiMessage.CamAiImageList(alarm_cache)
                                writer_queue.put(batchmessage)

                            # Send to notifications queue also, it will
                            # try to find a frame with the 'best' face
                            # in the first batch and use that for notification instead
                            # of blindly using the first frame
                            if notified_once is False:
                                face_sampling_frequency = 9
                                sampled_images = alarm_cache[None:None:face_sampling_frequency]
                                # matchesarray contains matches for object of
                                # interest in each of the sampled images
                                matchesarray = []
                                logger.debug (f"{name}: Sampled {len(sampled_images)} alarm frames ")
                                # If facedetection is enabled
                                for sample in sampled_images:
                                    results, should_quit = self.do_detection([sample])
                                    if should_quit is True:
                                        do_observe = False
                                        break

                                    r = results[0]
                                    matched_objects_dict = CamAiDetection.get_watched_matches(r, objects_of_interest)
                                    # Pass all the dicts directly in an array to notifier
                                    matchesarray.append(matched_objects_dict)

                                    #for object_index in matched_objects_dict:
                                    #    # TODO: Short term hack while refactoring is underway
                                    #    #if object_index is CamAiDetection.Person_Index:
                                    #    #Let notifier deal with this as user requested all these objects tracked
                                    #    # Why aren't we passing the dict directly??
                                    #    matchesarray.append(matched_objects_dict[object_index]['matched_records'])

                                    # TODO: Add car recognition gate here
                                    if self.facedetection is True or self.annotation is True:
                                        if self.annotation is True:
                                            sample = CamAiDetection.annotate_objects_in_frame(
                                                sample , r['rois'], r['masks'],
                                                r['class_ids'], r['scores'])
                                    # Only one frame detection suffices for
                                    # notification purposes when facedetection is off
                                    else:
                                        break

                                # TODO: Functionize better to deal with this nested break mess
                                if should_quit is True:
                                    break

                                try:
                                    message = CamAiMessage.CamAiNotifyMsg(
                                        sampled_images, matchesarray, detect_time, self.name)
                                    camera_notification_queue.put(message)
                                    notified_once = True
                                except queue.Full:
                                    logger.warning(f"{name}: Could not send a notification message")

                        # TODO: Functionize better to deal with this nested break mess
                        if should_quit is True:
                            break

                # Record Only Mode
                elif (self.mode == CamAiCameraMode.record_only.name):
                    # In this mode, reader sends the frames directly to writer
                    # So not much to do other than wait for quit signal
                    logger.debug(f"{name}: Checking for quit request in record_only mode")
                    if self.check_for_quit_request(2):
                        break
                else:
                    logger.error(f"{name}: Unknown camera mode configured: {self.mode}")
                    # Wait for quit instead of spinning and wasting cycles
                    #logger.warning(f"{name}: Checking for quit request")
                    if self.check_for_quit_request(2):
                        break


            except KeyboardInterrupt:
                logger.warning(f"{name}: Got a keyboard interrupt, exiting")
                break

            # Check if manager wants us to quit
            # TODO: Re evaluate this quit check once all modes are
            # brought back to support, this might be unnecessary overhead
            # if all modes are already checking
            logger.debug(f"{name}: Checking for quit request in while")
            if self.check_for_quit_request():
                break

        # Exit processing

        # Flush object time series to disk
        if self.log_object_timeseries is True:
            self.update_object_timeseries(detect_time=None, matches=None, write=True)

        # Stop the reader thread first, as it's the source to the queues
        reader.stop()

        # Let notifier know it's time to quit
        quitmessage = CamAiMessage.CamAiQuitMsg()
        camera_notification_queue.put(quitmessage)

        # Now wait on reader to return
        reader.join(10)

        # Ask writer to quit via queue in case it's a process
        # even in direct mode, reader should send this quit also
        quitmessage = CamAiMessage.CamAiQuitMsg()
        writer_queue.put(quitmessage)

        # Set it to stop
        writer.stop()

        # Stop the threads and wait for them to finish, this will hopefully
        # give notifier time to quit as we are waiting
        writer.join(10)

        if self.multiprocessing_detector is True:
            camera_detect_queue.close()
            camera_response_queue.close()
            camera_oob_queue.close()

        if self.multiprocessing_reader is True:
            reader_queue.close()

        if self.multiprocessing_writer is True:
            writer_queue.close()

        if self.multiprocessing_notifier is True:
            camera_notification_queue.close()

        if DEBUG is True:
            logger.warning(f"{name}: Queue depth history: detect: {detect_qsize_monitor}\n response: {response_qsize_monitor}\n read:{read_qsize_monitor}\n write:{write_qsize_monitor}")

        os.sync()
        logger.warning(f"{name}: Returning")
        return

    # Returns True if we need to exit, False otherwise
    # If a non zero wait is specified, it becomes a blocking get
    def check_for_quit_request(self, wait=0):
        name = "Observer: " + self.name
        should_quit = False
        # Check if manager wants us to quit
        # TODO: Re evaluate this blocks position once all modes are
        # brought back to support, this might be unnecessary overhead
        # for some modes
        try:
            if wait == 0:
                oobmessage = self.oob_queue.get(False)
            else:
                oobmessage = self.oob_queue.get(True, wait)

            logger.warning(f"{name}: Got an OOB message")
            if oobmessage.msgtype == CamAiMessage.CamAiMsgType.quit:
                logger.warn(f"{name}: Got a quit message from Manager")
                should_quit = True
            else:
                logger.warn(f"{name}: Unhandled OOB message type")
        except queue.Empty:
            logger.debug(f"{name}:No OOB message to process")
            pass

        return should_quit

    # Returns an array of images, and if caller should quit
    def get_images_from_camera(self, number_of_images=1):
        images = []
        should_quit = False
        name = "Observer: " + self.name

        for image_num in range(number_of_images):
            try:
                message = self.reader_queue.get(True)
                if message.msgtype == CamAiMessage.CamAiMsgType.image:
                    image = message.msgdata
                    if np.any(image):
                        # TODO: Should be config file option
                        if self.rotation != 0:
                            # 2x faster than cv2 for 90/270 degrees
                            image = imutils.rotate_bound(image, self.rotation)
                        images.append(image)
                elif message.msgtype == CamAiMessage.CamAiMsgType.quit:
                    should_quit = True
                    logger.warning(f"{name}: Quit message received, observer will stop")
                    break
                else:
                    logger.warning(f"{name}: Unexpected message type received")

            except queue.Empty:
                logger.warning(f"{name}: Reader queue is empty ")
                continue

        return images, should_quit

    # Returns a detector response, TODO: make this the array of results instead
    def do_detection(self, detect_images):
        name = "Observer: " + self.name
        resultmessage = None
        results = None
        should_quit = False
        molded_images = []
        resize_meta = []


        for image in detect_images:
            molded_image, scale, padding = self.detection.resize_image(image)
            molded_images.append(molded_image)
            resize_meta.append([scale, padding])

        try:
            message = CamAiMessage.CamAiDetectImage(self.handle, molded_images)
            self.detect_queue.put(message)
            # logger.debug(f"{name}: Sent message to Detector queue with handle {self.handle}")
        except queue.Full:
            logger.warning(f"{name}: Detector queue is full")

        try:
            # logger.debug(f"{name}: Waiting for response from detector")
            resultmessage = self.response_queue.get(True)
            # logger.debug(f"{name}: Got response back from detector")
        except queue.Empty:
            logger.error(f"{name}: Blocking response back is empty")
        except KeyboardInterrupt:
            should_quit = True
            logger.warning(f"{name}: Got a keyboard interrupt while waiting for detector response")

        if resultmessage is None:
            logger.error(f"{name}: Item returned from detector is None, shouldn't happen, debug further")

        # logger.warning(f"Results : {results}")
        # Results : [{'rois': array([[328,   1, 505, 239],
        #                           [599, 858, 619, 885],
        #                           [336, 306, 394, 361],
        #                           [366, 659, 395, 686]], dtype=int32),
        #            'class_ids': array([ 3, 33, 59,  1], dtype=int32),
        #            'scores': array([0.9970293, 0.9864123, 0.9791309, 0.9220377], dtype=float32),
        #            'masks': array([[[False, False, False, False],
        #                             [False, False, False, False],
        #                             [False, False, False, False],
        #                             ...,
        #        ...,
        #        [False, False, False, False],
        #        [False, False, False, False],
        #        [False, False, False, False]]])}],
        #        Returns a list of dicts, one dict per image. The dict contains:
        #        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        #        class_ids: [N] int class IDs
        #        scores: [N] float probability scores for the class IDs
        #        masks: [H, W, N] instance binary masks
        #        padding = [(0, 0), (y_pos, x_pos), (h+y_pos,w+x_pos), (h+2*y_pos,w+2*x_pos)]
        #        Scale of resize is 0.4444444444444444, float: 2.25 int: 2
        #        Padding is [(0, 0), (504, 0), (792, 2304), (1296, 2304)],0
        #        shape of r['rois'] is (2, 4), vals = [[577 330 593 343]
        #                                              [310 510 441 562]]
        elif resultmessage.msgtype == CamAiMessage.CamAiMsgType.detectresult:
            results = resultmessage.msgdata['result']
            for result in results:
                # Are there any ROIs to process?
                # We might want to use these ROIs to do face detection
                # or ptz etc
                if np.shape(result['rois'])[0] > 0:
                    result['rois'] = np.multiply(
                        result['rois'],
                        (1 / scale)).astype(int)
                    # Y depadding
                    # r['rois'][:,0] = np.subtract(r['rois'][:,0], padding[1][0]).astype(int)
                    # r['rois'][:,2] = np.subtract(r['rois'][:,2], padding[1][0]).astype(int)
                    # X depadding
                    # r['rois'][:,1] = np.subtract(r['rois'][:,1], padding[1][1]).astype(int)
                    # r['rois'][:,3] = np.subtract(r['rois'][:,3], padding[1][1]).astype(int)

                    # Merging the above depadding into two passes
                    result['rois'][:, [0, 2]] = np.subtract(
                        result['rois'][:, [0, 2]],
                        padding[1][0]).astype(int)
                    result['rois'][:, [1, 3]] = np.subtract(
                        result['rois'][:, [1, 3]],
                        padding[1][1]).astype(int)

                if self.annotation is True:
                    # If there are any objects found
                    if np.shape(result['masks'])[2] > 0:
                        result['masks'] = scipy.ndimage.zoom(
                            result['masks'], zoom=[1/scale, 1/scale, 1], order=0)
                        logger.debug(f"{name}: Shape of result['masks'] after zoom is {np.shape(result['masks'])} ")
                        result['masks'] = result['masks'][
                            padding[1][0]: padding[2][0],
                            padding[1][1]: padding[2][1]]
                        logger.debug(f"{name}: Shape of result['masks'] after depadding is {np.shape(result['masks'])} ")

        elif resultmessage.msgtype == CamAiMessage.CamAiMsgType.quit:
            should_quit = True
            logger.warning(f"{name}: Quit message received, observer will stop")
        else:
            logger.error(f"{name}: Unexpected message type received")

        return results, should_quit

if __name__ == '__main__':
    pass
