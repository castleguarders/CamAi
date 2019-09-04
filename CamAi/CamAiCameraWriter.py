import os
import shutil
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

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import CamAiMessage
#import CamAiMessage
#import CamAiDetection
#import CamAiUtils

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
#formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
#formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
formatter = logging.Formatter('%(asctime)s:%(name)s:%(funcName)s:%(lineno)d:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

#file_handler = logging.FileHandler('CamAiCameraWriter.errorlog')
#file_handler.setFormatter(formatter)
#file_handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
#stream_handler.setLevel(logging.DEBUG)
#stream_handler.setLevel(logging.WARNING)
stream_handler.setLevel(logging.ERROR)

#logger.addHandler(file_handler)
logger.addHandler(stream_handler)

DEBUG = False

# Give main thread time to be ready and waiting for inference
# so frames don't get lost in the beginning
Camera_Startup_Wait = 20
priming_reconnect_attempts = 3
priming_read_attempts = 10
priming_connect_wait = 4
priming_read_wait = 1

Capture_Watchdog_Timer = 60  # One minutes

FOURCC = cv.VideoWriter_fourcc(*'mp4v')
# FOURCC = cv.VideoWriter_fourcc(*'h264')
# FOURCC = cv.VideoWriter_fourcc(*'h265')
# FOURCC = cv.VideoWriter_fourcc(*'X264')
# FOURCC = cv.VideoWriter_fourcc(*'xvid')
# FOURCC = cv.VideoWriter_fourcc(*'avc1')

# Video_File_Extension = '.h264'
Video_File_Extension = '.mp4'


def write_images_to_video(images, video_filename="./video.mp4", fps=24):
    if images is None or len(images) == 0:
        logger.warning("No images passed, no video written")
        return

    (h, w) = images[0].shape[:2]
    logger.warn(
        "Image shape is height {}, width {}".format(h, w))

    video_log = cv.VideoWriter(
        video_filename, FOURCC, fps, (w, h), True)
    if video_log.isOpened() is not True:
        logger.error("video file could not be opened file: {} w: {} h: {} ".format(
            video_filename, w, h))

    for image in images:
        video_log.write(image)

    os.sync()


class CamAiCameraWriter(object):

    def __init__(
                self,
                config,
                ):

        self.config = config
        self.vfdb = None

    def start(self):
        cname = "Writer: " + self.config.name
        if self.config.multiprocessing_writer is True:
            self.writer = Process(
                target=self._write_video, args=([]), name=cname)
        else:
            self.writer = threading.Thread(
                target=self._write_video, args=([]), name=cname)
            self.writer.do_write= True

        logger.warn("{} : Starting".format(cname))
        self.writer.start()

    def stop(self):
        cname = "Writer: " + self.config.name
        if self.config.multiprocessing_writer is False:
            self.writer.do_write = False
        logger.warn("{} : Stopping".format(cname))

    def join(self, waittime=10):
        cname = "Writer: " + self.config.name
        self.writer.join(waittime)
        logger.warn("{} : Joining".format(cname))

    def get_video_log_filename(self, now):
        filename = (self.config.name + '_' + (str)(now.year) + '_' + (str)
            (now.month) + '_' + (str)(now.day) + '_' + (str)(now.hour) + '_' + (str)
            (now.minute) + '-' + (str)(now.second) + '_' + (str)(now.microsecond) +
            (str)(random.randint(0, 100)) + Video_File_Extension)
        video_log_filename = os.path.join(self.config.mydir, filename)

        return video_log_filename

    # This gets the history of files written to disk
    def get_video_database(self):
        name = "Writer: " + self.config.name
        parquet_file = os.path.join(self.config.mydir, self.config.name + '_videodb.parquet')
        if os.path.isfile(parquet_file):
            dbtable = pq.read_table(parquet_file)
            vfdb = dbtable.to_pandas()
            if vfdb is None:
                logger.error("{} : vfdb is None after loading {}".format(
                    name, parquet_file))
            else:
                logger.debug(
                    "{} : vfdb types: {}".format(
                        name, vfdb.dtypes))
        else:
            # First time, so create the db dataframe
            col_names = ['start_time',
                         'vfile_name',
                         'end_time',
                         'event',
                         'deleted',
                         'spaceused'
                        ]
            vfdb = pd.DataFrame(columns=col_names)
            vfdb['start_time'] = pd.to_datetime(vfdb['start_time'])
            vfdb['end_time'] = pd.to_datetime(vfdb['end_time'])
            vfdb['vfile_name'].astype('str', copy=False)
            vfdb['event'].astype('bool', copy=False)
            vfdb['deleted'].astype('bool', copy=False)
            vfdb['spaceused'].astype('int64', copy=False)
            vfdb = vfdb.set_index('start_time')

        self.vfdb = vfdb
        logger.debug("{} : vfdb entries: {}".format(name, self.vfdb))

    def update_video_database(self, start_time, vfile_name, end_time, event):
        if self.vfdb is None:
            self.get_video_database()

        name = "Writer: " + self.config.name
        vfdb = self.vfdb
        deleted = False

        try:
            space_used = os.path.getsize(vfile_name)
            vfdb.loc[start_time] = [vfile_name, end_time, event, deleted, space_used]
        except FileNotFoundError:
                logger.warning("{}: File missing, zero frames maybe?".format(name))

        logger.debug("{} : ===============================================")
        logger.debug("{} : start_time: {}, vfile_name : {}, end_time: {}, event: {}, deleted:{}".
                    format(name, start_time, vfile_name, end_time, event, deleted))
        logger.debug("{} : ===============================================")
        logger.debug("{} : vfdb entries: {}".format(name, self.vfdb))
        logger.debug("{} : ===============================================")

        parquet_file = os.path.join(self.config.mydir, self.config.name + '_videodb.parquet')
        if os.path.isfile(parquet_file):
            logger.debug ("{}:File exists, will overwrite".format(name))
        vfdbtable = pa.Table.from_pandas(vfdb)
        pq.write_table(vfdbtable, parquet_file)

        self.vfdb = vfdb

    def prune_video_database (self):
        if self.vfdb is None:
            self.get_video_database()
        name = "Writer: " + self.config.name

        vfdb = self.vfdb

        now = datetime.datetime.now()
        prune_older_than_hours = datetime.timedelta(days=self.config.deleteafterdays,
                                                    hours=now.hour, minutes=now.minute,
                                                    seconds=now.second,
                                                    microseconds=now.microsecond)
        prune_from = now - datetime.timedelta(days=1*365,
                                              hours=now.hour,
                                              minutes=now.minute,
                                              seconds=now.second,
                                              microseconds=now.microsecond)
        prune_to = now - prune_older_than_hours
        logger.warning(f"{name}: Pruning from {prune_from} to {prune_to}")

        parquet_file = os.path.join(self.config.mydir, self.config.name + '_videodb.parquet')
        total, used, free = shutil.disk_usage(parquet_file)
        percentage_free = int((free*100)/total)
        logger.debug(f"{name}: Disk free = {percentage_free}%, total: {int(total/(1024*1024))} MB, used: {int(used/(1024*1024))} MB, free: {int(free/(1024*1024))} MB")

        # TODO: Get threshold from configuration file
        if percentage_free <= self.config.deletethreshold:
            #logger.warn("Starting vfdb is {}".format(vfdb))
            vfdb_prune = vfdb[prune_from: prune_to]
            # Delete files with events if freespace lower than events threshold
            if percentage_free <= self.config.deleteeventsthreshold:
                vfdb_prune = vfdb_prune[vfdb_prune['event'] == False].sort_index()
            files_to_prune = vfdb_prune[['vfile_name']]

            total_freed = 0
            for file_tobe_deleted in files_to_prune['vfile_name']:
                try:
                    space_freed = os.path.getsize(file_tobe_deleted)
                    total_freed += space_freed
                    logger.warn(f"{name}: Deleting file: {file_tobe_deleted}, freed: {int(space_freed/(1024*1024))} MB")
                    os.remove(file_tobe_deleted)
                except FileNotFoundError:
                    #logger.exception("{}: File {} not found, might have been manually deleted?".format(
                    logger.debug(f"{name}: File {file_tobe_deleted} not found, might have been manually deleted?")

            logger.warn(f"{name}: Total Freed : {int(total_freed/(1024*1024))} MB")

            #vfdb_prune['deleted'].loc[vfdb_prune['deleted'] == False] = True
            vfdb.update(vfdb_prune)
            #logger.warn("vfdb: index is {}".format(vfdb.loc[vfdb_prune.index]['deleted']))
            logger.debug(f"{name} : vfdb entries after update: {vfdb}")

            if os.path.isfile(parquet_file):
                logger.debug(f"{name}:File exists, will overwrite")

            vfdbtable = pa.Table.from_pandas(vfdb)
            pq.write_table(vfdbtable, parquet_file)
            #logger.warn(f"Updated vfdb is {vfdb}")

        self.vfdb = vfdb

    # @profile
    def _write_video(self):

        writer_queue = self.config.writer_queue
        name = "Writer: " + self.config.name

        file_rotate_date = datetime.datetime.now()
        video_log_filename = self.get_video_log_filename(file_rotate_date)
        video_log = None
        h = 0
        w = 0

        # Profiling vars
        cum_gettime = 0
        cum_writetime = 0
        num_frames_written = 0
        num_gets = 0
        events_in_file = False

        writer = threading.currentThread()

        do_write = True  # Process get keyboard exits, cannot be told directly by parent

        while do_write is True:
            if (self.config.multiprocessing_writer is False):
                # TODO: In progress exit notifications only through queues
                # do_write =  writer.do_write
                pass

            try:
                try:
                    starttime = time.perf_counter()
                    message = writer_queue.get(True)
                    # Profiling time spent on gets
                    cum_gettime += (time.perf_counter() - starttime)
                    num_gets += 1
                    if message.msgtype == CamAiMessage.CamAiMsgType.image:
                        # logger.warn("Got a single image")
                        frame = message.msgdata
                        if np.any(frame):
                            starttime = time.perf_counter()
                            if (video_log is None):
                                (h, w) = frame.shape[:2]
                                logger.debug(
                                    "{}: Image shape is height {}, width {}".format(
                                        name, h, w))
                                video_log = cv.VideoWriter(
                                    video_log_filename, FOURCC, self.config.fps, (w, h), True)
                            # Profiling time spent on writing
                            (h, w) = frame.shape[:2]
                            rc = video_log.write(frame)
                            # logger.debug("{}: Image shape is height {}, width {}, rc {}".format(name, h, w, rc))
                            num_frames_written += 1
                            cum_writetime = (time.perf_counter() - starttime)
                        else:
                            logger.warn(
                                "{}: No frame in image message type!".format(name))
                    elif message.msgtype == CamAiMessage.CamAiMsgType.imagelist:
                        events_in_file = True
                        logger.debug(
                            "{}: Got an imagelist of length {}".format(
                                name, len(message.msgdata)))
                        for frame in message.msgdata:
                            starttime = time.perf_counter()
                            if (video_log is None):
                                (h, w) = frame.shape[:2]
                                logger.debug(
                                    "{}: Image shape is height {}, width {}".format(
                                        name, h, w))
                                video_log = cv.VideoWriter(
                                    video_log_filename, FOURCC, self.config.fps, (w, h), True)
                                if video_log.isOpened() is not True:
                                    logger.error("{}: video log could not be opened file: {} w: {} h: {} ".format(
                                        name, video_log_filename, w, h))
                            # Profiling time spent on writing
                            video_log.write(frame)
                            num_frames_written += 1
                            cum_writetime = (time.perf_counter() - starttime)
                        # So we don't need to reference frame later and we don't
                        # use stale info in case resolution changes
                        (h, w) = frame.shape[:2]
                        # See if explicit dels speed up GC any
                        #del message.msgdata
                        #del message
                    elif message.msgtype == CamAiMessage.CamAiMsgType.quit:
                        logger.warn(
                            "{}: Quit message recieved: writer will stop".format(name))
                        break
                    else:
                        logger.warn(
                            "{}: Unknown message type received by writer".format(name))
                        pass
                except AttributeError:
                    logger.exception(
                        "{}: AttributeError, not expected, debug further".format(name))
                except queue.Empty:
                    logger.exception("{}: Queue is empty".format(name))
                    break

                # Check if it's a new hour, log to a new file if it is
                now = datetime.datetime.now()
                #if (file_rotate_date.hour != now.hour):
                if (file_rotate_date.hour != now.hour) or (file_rotate_date.minute + 30) <= now.minute:
                    if (video_log is not None):
                        video_log.release()
                    os.sync()

                    self.update_video_database(file_rotate_date,
                                               video_log_filename,
                                               now, events_in_file)

                    # Free up space based on policy configured
                    if self.config.deletepolicyenabled is True:
                        self.prune_video_database()

                    events_in_file = False
                    video_log_filename = self.get_video_log_filename(now)
                    logger.debug(
                        "{}: Rotating log file to {}, {}, {}".format(
                            name, video_log_filename, file_rotate_date, now))
                    logger.debug(
                        "{}: Image height {}, width {}".format(name, h, w))
                    video_log = cv.VideoWriter(video_log_filename,
                                               FOURCC,
                                               self.config.fps,
                                               (w, h),
                                               True
                                               )
                    if video_log.isOpened() is not True:
                        logger.error("{}: video log could not be opened file: {} w: {} h: {} ".format(
                            name, video_log_filename, w, h))
                    file_rotate_date = now

            # This is only necessary when run as a process
            except KeyboardInterrupt:
                logger.warn(
                    "{}: Keyboard interrupt will wait for a Quit message for a clean exit".format(name))
                pass

        # Update video file database
        now = datetime.datetime.now()
        self.update_video_database(file_rotate_date,
                                   video_log_filename,
                                   now, events_in_file)

        # Free up space based on policy configured
        if self.config.deletepolicyenabled is True:
            self.prune_video_database()

         # Dump some profiling stats to understand our bottlenecks better
        if DEBUG is True:
            self.dump_stats(name, cum_gettime, cum_writetime, num_frames_written, num_gets)

        if (video_log is not None):
            logger.warn("{}: Released video_log in writer".format(name))
            video_log.release()

        os.sync()
        # if (self.config.multiprocessing_writer is True):
        #    writer_queue.close()
        logger.warn("{}: Returning from writer {}".format(name, writer.name))
        return


    def dump_stats(self, name, cum_gettime, cum_writetime, num_frames_written, num_gets):
        # Dump some profiling stats to understand our bottlenecks better
        logger.warn(
            "{}: ============================================================================".format(name))
        logger.warn("{}: Number of gets processed: {}, ".format(name, num_gets))
        logger.warn(
            "{}: Number of frms processed: {}, ".format(
                name, num_frames_written))
        logger.warn("{}: Cumulative time on gets: {:.2f}".format(name, cum_gettime))
        logger.warn("{}: Cumulative time on frms: {:.2f}".format(name, cum_writetime))
        if num_gets > 0:
            logger.warn("{}: Average milliseconds per get: {:.2f}".format(
                            name, 1000 * (cum_gettime / num_gets)))
        if num_frames_written > 0:
            logger.warn("{}: Average milliseconds per frm: {:.2f}".format(
                name, 1000 * (cum_writetime/num_frames_written)))
        logger.warn(
            "{}: ============================================================================".format(name))



if __name__ == '__main__':
    pass
