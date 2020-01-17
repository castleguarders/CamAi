from enum import Enum
import cProfile
import threading


class CamAiMsgType(Enum):
    notification = 100
    configuration = 120
    image = 200
    imagelist = 300
    detectimage = 400
    detectresult = 500
    ping = 600
    pingresponse = 700
    quit = 900

# Can also declare this as
# CamAiMsgType = Enum('CamAiMsgType', 'notification image imagelist')


# Base class for messages exchanged on CamAi queues
class CamAiMsg (object):
    msgtype = None
    msgdata = None

    def __init__(
                self,
                msgtype,
                msgdata,
                ):

        self.msgtype = msgtype
        self.msgdata = msgdata


# Data classes would be nice, but only in 3.7 and above
class CamAiNotifyMsg(CamAiMsg):
    def __init__(self, image, objectsdetected, timestamp, cameraconfig):
        msgtype = CamAiMsgType.notification
        msgdata = {'image': image,
                   'objects detected': objectsdetected,  # List of Dicts
                   'timestamp': timestamp,
                   'cameraconfig': cameraconfig
                   }
        super().__init__(msgtype, msgdata)

# class CamAiConfigMsgType(Enum):
#     update = 100
#     pause_emails = 120
#     pause_verbal = 121
#     pause_sms = 122
#     restart = 900
#
# # configmsg is managed separately
# class CamAiConfigurationMsg(CamAiMsg):
#     def __init__(self, cameraname, configmsg):
#         msgtype = CamAiMsgType.configuration
#         msgdata = {
#                    'configmsgtype': configmsgtype,
#                    'cameraname': cameraname
#                    }
#         super().__init__(msgtype, msgdata)

class CamAiDetectImage(CamAiMsg):
    def __init__(self, camera_handle, image, detection_type='maskrcnn'):
        msgtype = CamAiMsgType.detectimage
        msgdata = {'camera_handle': camera_handle,
                   'image': image,
                   'detection_type': detection_type
                   }
        super().__init__(msgtype, msgdata)


class CamAiDetectResult(CamAiMsg):
    def __init__(self, camera_handle, result):
        msgtype = CamAiMsgType.detectresult
        msgdata = {'camera_handle': camera_handle,
                   'result': result
                   }
        super().__init__(msgtype, msgdata)

class CamAiPing(CamAiMsg):
    def __init__(self):
        msgtype = CamAiMsgType.ping
        msgdata = 'ping'
        super().__init__(msgtype, msgdata)

class CamAiPingResponse(CamAiMsg):
    def __init__(self, name, response):
        msgtype = CamAiMsgType.pingresponse
        msgdata = {'name': name,
                   'response': response
                   }
        super().__init__(msgtype, msgdata)


class CamAiImage(CamAiMsg):
    def __init__(self, image):
        msgtype = CamAiMsgType.image
        msgdata = image
        super().__init__(msgtype, msgdata)


class CamAiImageList(CamAiMsg):
    def __init__(self, imagelist):
        msgtype = CamAiMsgType.imagelist
        msgdata = imagelist
        super().__init__(msgtype, msgdata)


class CamAiQuitMsg(CamAiMsg):
    def __init__(self):
        msgtype = CamAiMsgType.quit
        msgdata = 'quit'
        super().__init__(msgtype, msgdata)


class ProfiledThread(CamAiQuitMsg):
    profiler_fileprefix = "profile_cam_ai_"

    # Overrides threading.Thread.run()
    def run(self):
        profiler = cProfile.Profile()
        try:
            return profiler
        finally:
            t = threading.currentThread()
            file_name = "{}_{}.profile".format(
                self.profiler_fileprefix, t.ident)
            print(
                "Dumping profile file {} for thread {}".format(
                    file_name, t.name))
            print(file_name)

    def set_profile_file(self, fileprefix):
        self.profiler_fileprefix = fileprefix
        print("Set profile prefix to {}".format(self.profiler_fileprefix))


if __name__ == '__main__':
    mc = CamAiMsgType(CamAiMsgType.notification)
    for messagetype in CamAiMsgType:
        print(messagetype)

    test1 = ProfiledThread()
    test1.set_profile_file("blahblah")
    test1.run()
