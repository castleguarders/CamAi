import os
from enum import Enum
import logging
import toml

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
stream_handler.setLevel(logging.DEBUG)

#logger.addHandler(file_handler)
logger.addHandler(stream_handler)

DEBUG = False

class CamAiCameraMode(Enum):
    detect_and_record_timelapse = 100
    detect_and_record_everything = 200
    detect_only = 300
    record_only = 400

Default_Notify_Startup_Wait = 8 # minutes
Default_Detection_Threshold = 0.98
Default_ROI_Match_Threshold = 75
Default_Watch_Timerange_Start = 8 # minutes ago, ideally less than startup_notify_wait
Default_Watch_Timerange_End = 2 # minutes ago
Default_Watch_Continuous_Mode = False
Default_Objects_Vars =     {'detection_threshold': Default_Detection_Threshold,
                            'instance_match_threshold': Default_ROI_Match_Threshold,
                            'instance_watch_timerange_start': Default_Watch_Timerange_Start,
                            'instance_watch_timerange_end': Default_Watch_Timerange_End,
                            'instance_watch_continuous_mode': Default_Watch_Continuous_Mode,
                            'notify_startup_wait': Default_Notify_Startup_Wait,
                            'instancedetection': False
                            }
Default_Objects_to_Watch = {'person': Default_Objects_Vars,
                            'car':  Default_Objects_Vars,
                            'motorcycle':  Default_Objects_Vars,
                            'truck':  Default_Objects_Vars,
                            'bus':   Default_Objects_Vars,
                            'bicycle':   Default_Objects_Vars
                            }

CAMVARS = {'name': 'Camera Name',
           'url': '0',
           'mode': CamAiCameraMode.detect_and_record_timelapse.name,
           'subdir': None,
           'rotation': 0,
           'readbuffer': 16*1920*1080,
           'fps': 25,
           'watchdogenable': True,
           'watchdogtimer': 60,
           'annotation': False,
           'showlivevideo': False,
           'facedetection': False,
           'maxreadqueue': 32,
           'detectionrate': 25,
           'detecttrackseconds': 30,
           'deletepolicyenabled': True,
           'deletethreshold': 60,
           'deleteeventsthreshold':80,
           'deleteafterdays':7,
           'log_object_timeseries':True,
           'objects_of_interest': Default_Objects_to_Watch
           }

MANAGERVARS = {'basedir': "./videos/",
               'numdetectors': 1,
               'pipelineresize': True,
               'singledetectorqueue': True,
               'defaultmaxqueue': 32,
               'multiprocessing_observer': False,
               'multiprocessing_detector': False,
               'multiprocessing_reader': False,
               'multiprocessing_writer': False,
               'multiprocessing_notifier': True,
               'multiprocessing_viewer': False,
               'multiprocessing_trainer': True
               }

class CamAiConfig(object):

    def __init__(
                self,
                config_file
                ):

        if os.path.isfile(config_file) is False:
            raise ValueError(
                "Configuration file {} does not exist".format(config_file))

        self.config_file = config_file
        self.config = toml.load(config_file)

    def get_manager_options(self):
        return self.config["manager options"]

    def get_detector_options(self):
        return self.config["detector options"]

    def get_cameras(self):
        return self.config["camera"]

    def get_email_sender(self):
        try:
            return self.config["email sender"]
        except KeyError:
            logger.warning("Email Sender section not in configuration file")
            return {}

    def get_email_recepients(self):
        try:
            return self.config["email recepient"]
        except KeyError:
            logger.warning("Email Recepients section not in configuration file")
            return []


    def get_basedir(self):
        return self.config["manager options"]["basedir"]

def describe_var(root, prefixstr):
    for node in root:
        if isinstance(root[node], collections.Mapping):
            print("{}Iterable Variable {}: ".format(prefixstr, node))
            prefixstr2 = prefixstr + "\t"
            describe_var(root[node], prefixstr2)
        else:
            print("{}Variable {}: Value: {}".format(prefixstr, node, root[node]))


# Unit tests
if __name__ == '__main__':

    # Non existent file
    #camai_config = CamAiConfig("thisfilesdoesnotexist.toml")

    # File exists, but is we don't have access
    #camai_config = CamAiConfig("/etc/shadow")

    # This file is real and exists, but is not toml format
    #camai_config = CamAiConfig("./Readme.txt")
    import sys
    import collections
    try:
        if sys.argv[1]:
            configf = sys.argv[1]
    except:
        configf = "./racam.toml"

    print("Toml input file is: {}".format(configf))
    camai_config = CamAiConfig(configf)

    camaiconfig = camai_config.config

    print("Toml dictionary is: {}".format(camaiconfig))

    #cameras = camaiconfig["camera"]
    cameras = camai_config.get_cameras()
    #print("cameras is : {}".format(cameras))

    for camera in cameras:
        describe_var(camera, "")

    #recepients = camaiconfig["Email Recepient"]
    recepients = camai_config.get_email_recepients()
    #print ("\nrecepients: {}".format(recepients))
    for recepient in recepients:
        print("\nRecepient \n\tname:{}\n\temail_address:{}\n\t".format(
            recepient['name'], recepient['email_address']))

    #email_sender = camaiconfig["Email Sender"]
    email_sender = camai_config.get_email_sender()
    for key in email_sender:
        print("{} :  {}\n".format(key, email_sender[key]))

    #storage_configuration = camaiconfig["Storage Configuration"]
    #for key in storage_configuration:
    #    print("{} :  {}\n".format(key, storage_configuration[key]))

    basedir = camai_config.get_basedir()

    print("Base directory is {}\n".format(basedir))

    #export test
    #with open("tracam.toml", 'w') as dumpfile:
    #    toml.dump(camaiconfig, dumpfile)

    #print("Toml dictionary is: {}".format(camaiconfig))
