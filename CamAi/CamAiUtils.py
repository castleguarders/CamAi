import os
import threading
from multiprocessing import Process
import logging
import cProfile

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
#stream_handler.setLevel(logging.DEBUG)
#stream_handler.setLevel(logging.WARNING)
stream_handler.setLevel(logging.ERROR)

#logger.addHandler(file_handler)
logger.addHandler(stream_handler)

DEBUG = False


def increase_my_priority(increase):
    # Set the reader process to high priority to reduce frame processing latency,
    # we lose frames under load otherwise, hopefully this will reduce the lossage
    # TODO Handle windows if necessary
    curpriority = os.getpriority(os.PRIO_PROCESS, 0)
    logger.debug("Reader default priority is {}".format(curpriority))
    try:
        os.setpriority(os.PRIO_PROCESS, 0, (curpriority - increase))
        logger.warn("Increased priority to: {}".format(curpriority-increase))
    except PermissionError:
        logger.warn(
            "Inadequate permissions to increase the reader priority,consider running with sudo, priority is: {}".
            format(curpriority))

    newpriority = os.getpriority(os.PRIO_PROCESS, 0)
    logger.debug("Reader new priority is {}".format(newpriority))


class ProfiledThread(threading.Thread):
    profiler_fileprefix = "profile_cam_ai_"

    # Overrides threading.Thread.run()
    def run(self):
        profiler = cProfile.Profile()
        try:
            return profiler.runcall(threading.Thread.run, self)
        finally:
            t = threading.currentThread()
            file_name = "{}_{}.profile".format(
                self.profiler_fileprefix, self.ident)
            logger.warn(
                "Dumping profile file {} for thread {}".format(
                    file_name, t.name))
            profiler.dump_stats(file_name)

    def set_profile_file(self, fileprefix):
        logger.warn("Setting profile prefix to {}".format(fileprefix))
        self.profiler_fileprefix = fileprefix


class ProfiledProcess(Process):
    # Overrides Process.run()
    def run(self):
        profiler = cProfile.Profile()
        try:
            # return profiler.runcall(threading.Thread.run, self)
            return profiler.runcall(Process.run, self)
        finally:
            profiler.dump_stats('processprofile_%d.profile' % (self.ident,))
