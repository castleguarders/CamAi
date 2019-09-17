import logging
import toml
from subprocess import Popen
from subprocess import DEVNULL

# Required for WS-Discovery (Multicast / UDP based)
import re
import sys
from wsdiscovery.daemon import WSDiscovery
from wsdiscovery.scope import Scope
from wsdiscovery.qname import QName
from urllib.parse import urlparse
from urllib.parse import urlunparse
from urllib.parse import urlencode

# Required for ONVIF queries to discovered cameras
import onvif
from onvif import ONVIFCamera
import zeep
import asyncio, sys
import time, datetime

from . import CamAiConfig
#import CamAiConfig

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
stream_handler.setLevel(logging.WARNING)

#logger.addHandler(file_handler)
logger.addHandler(stream_handler)

STREAM_PLAYER = 'ffplay'
DEBUG = False


def discover_cameras(scope=None, types='NetworkVideoTransmitter', capture=None):
    wsd = WSDiscovery(capture=capture)
    wsd.start()
    namespace="http://www.onvif.org/ver10/network/wsdl"

    if not scope and not types:
        svcs = wsd.searchServices()
    elif scope and not types:
        # we support giving just one scope, for now
        svcs = wsd.searchServices(scopes=[Scope(scope)])
    elif not scope and types:
        # we support giving just one scope, for now
        probetypes = [QName(namespace, types)]
        svcs = wsd.searchServices(types=probetypes)
    else:
        # we support giving just one scope, for now
        probetypes = [QName(namespace, types)]
        svcs = wsd.searchServices(scopes=[Scope(scope)], types=probetypes)

    cameras = []
    for service in svcs:
        camera = {}

        url = urlparse(service.getXAddrs()[0])
        port = url.port
        if not port:
            port = 80

        camera['hostname'] = url.hostname
        camera['port'] = port
        camera['username'] = url.username
        camera['password'] = url.password

        #scopes =  service.getScopes()
        #camera['scopes'] = scopes

        cameras.append(camera)

    wsd.stop()

    if len(cameras) > 0:
        print("Found {} cameras".format(len(cameras)))
        for camera in cameras:
            print("    {}:{}".format(camera['hostname'], camera['port']))
    else:
        print("Did not find any cameras")


    return cameras

# Patch fix for a zeep issue
def zeep_pythonvalue(self, xmlvalue):
    return xmlvalue

zeep.xsd.simple.AnySimpleType.pythonvalue = zeep_pythonvalue

def get_camera_services(camera):
    mycam = ONVIFCamera(camera['hostname'] , camera['port'], camera['username'], camera['password'], '/usr/local/lib/python3.6/site-packages/wsdl')

    # Get List of Services
    devicemgmt_service = mycam.create_devicemgmt_service()
    servreq = devicemgmt_service.create_type('GetServices')
    servreq.IncludeCapability = True
    services = devicemgmt_service.GetServices(servreq)
    logging.warn ("Services : {}".format(services))
    for service in services:
        logging.warn ("Service: {} : ".format(service))

def get_device_service_capabilities(camera):
    mycam = ONVIFCamera(camera['hostname'] , camera['port'], camera['username'], camera['password'], '/usr/local/lib/python3.6/site-packages/wsdl')

    # Get List of Services
    devicemgmt_service = mycam.create_devicemgmt_service()
    servcapreq = devicemgmt_service.create_type('GetServiceCapabilities')
    service_caps = devicemgmt_service.GetServiceCapabilities(servcapreq)
    logging.warning ("Service Capabilities: {}".format(service_caps))
    for servicecap in service_caps:
        logging.warning ("Service capability: {} : ".format(servicecap))

def get_camera_capabilities(camera):
    mycam = ONVIFCamera(camera['hostname'] , camera['port'], camera['username'], camera['password'], '/usr/local/lib/python3.6/site-packages/wsdl')

    # Get List of Capabilities
    capabilities = mycam.devicemgmt.GetCapabilities()
    logging.debug ("Capabilities: {}".format(capabilities))
    for capability in capabilities:
        logging.debug ("Capability: {} : ".format(capability))

def get_access_policy(camera):
    mycam = ONVIFCamera(camera['hostname'] , camera['port'], camera['username'], camera['password'], '/usr/local/lib/python3.6/site-packages/wsdl')

    # Get List of Services
    devicemgmt_service = mycam.create_devicemgmt_service()
    accesspolicy = devicemgmt_service.GetAccessPolicy()
    logging.debug("Services : {}".format(accesspolicy))


def get_camera_time(camera):
    mycam = ONVIFCamera(camera['hostname'] , camera['port'], camera['username'], camera['password'], '/usr/local/lib/python3.6/site-packages/wsdl')

    # Get Hostname
    #resp = mycam.devicemgmt.GetHostname()
    #logging.debug ('My camera`s hostname: {}'.format(str(resp.Name)))
    dt = mycam.devicemgmt.GetSystemDateAndTime()
    tz = dt.TimeZone
    year = dt.UTCDateTime.Date.Year
    hour = dt.UTCDateTime.Time.Hour

    logging.warning ('My camera`s time: {}'.format(dt))

def set_camera_time(camera):
    mycam = ONVIFCamera(camera['hostname'] , camera['port'], camera['username'], camera['password'], '/usr/local/lib/python3.6/site-packages/wsdl')

    settime =  mycam.devicemgmt.create_type('SetSystemDateAndTime')
    settime.DateTimeType = 'Manual'
    settime.DaylightSavings = True
    settime.TimeZone = {'TZ': datetime.datetime.now(datetime.timezone.utc).astimezone().tzname()}
    utctime = datetime.datetime.utcnow()
    settime.UTCDateTime = {'Time': {'Hour': utctime.hour,
                                    'Minute': utctime.minute,
                                    'Second': utctime.second },
                           'Date': {'Year': utctime.year,
                                    'Month': utctime.month,
                                    'Day': utctime.day }
                          }
    '''
    'DateTimeType': 'Manual',
    'DaylightSavings': False,
    'TimeZone': {
        'TZ': 'CST-8'
    },
    'UTCDateTime': {
        'Time': {
            'Hour': 19,
            'Minute': 10,
            'Second': 29
        },
        'Date': {
            'Year': 2017,
            'Month': 12,
            'Day': 31
        }
    }'''
    response = mycam.devicemgmt.SetSystemDateAndTime(settime)
    dt = mycam.devicemgmt.GetSystemDateAndTime()

    logging.warning ('My camera`s time: {}'.format(dt))

def get_camera_name(camera):
    #print (f"Using camera {camera}")
    mycam = ONVIFCamera(camera['hostname'] , camera['port'], camera['username'], camera['password'], '/usr/local/lib/python3.6/site-packages/wsdl')

    # Get Hostname
    resp = mycam.devicemgmt.GetHostname()
    logging.debug ('My camera`s hostname: {}'.format(str(resp.Name)))
    return resp.Name


def get_media_profile_configuration(camera):
    '''
    A media profile consists of configuration entities such as video/audio
    source configuration, video/audio encoder configuration,
    or PTZ configuration. This use case describes how to change one
    configuration entity which has been already added to the media profile.
    '''

    # Create the media service
    mycam = ONVIFCamera(camera['hostname'] , camera['port'], camera['username'], camera['password'], '/usr/local/lib/python3.6/site-packages/wsdl')
    resp = mycam.devicemgmt.GetHostname()
    logging.debug ('My camera`s hostname: {}'.format(str(resp.Name)))

    media_service = mycam.create_media_service()

    #video_sources = media_service.GetVideoSources()
    #logging.debug ("Video Sources: {}".format(video_sources))

    # Get target profile
    media_profiles = media_service.GetProfiles()
    #media_profile = media_service.GetProfiles()[0]
    #logging.debug ("====================================================================")
    #logging.debug ("Media Profiles : {}".format(media_profiles))
    #logging.debug ("====================================================================")

    # Get all video encoder configurations
    #encoder_configurations_list = media_service.GetVideoEncoderConfigurations()

    obj = media_service.create_type('GetStreamUri')
    obj.StreamSetup = {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}}
    for profile in media_profiles:
        logging.debug ("====================================================================")
        logging.debug ("Media Profile : {}".format(profile))
        obj.ProfileToken = profile.token
        stream_uris = media_service.GetStreamUri(obj)
        snapshot_uris = media_service.GetSnapshotUri({'ProfileToken':profile.token})
        logging.debug ("====================================================================")
        logging.debug ("Stream URIs for this profile : {}".format(stream_uris))
        logging.debug ("Snapshot URIs for this profile : {}".format(snapshot_uris))

        #for configuration in  encoder_configurations_list:
        #    logging.debug ("====================================================================")
        #    logging.debug ("Video Configuration : {}".format(configuration))

        # Get video encoder configuration options
        #options = media_service.GetVideoEncoderConfigurationOptions({'ProfileToken':profile.token})
        #logging.debug ("====================================================================")
        #logging.debug ("VideoEncoder Options: {}".format(options))

    logging.debug ("====================================================================")


def is_ptz_supported(camera):
    mycam = ONVIFCamera(camera['hostname'] , camera['port'], camera['username'], camera['password'], '/usr/local/lib/python3.6/site-packages/wsdl')
    try:
        ptz = mycam.create_ptz_service()
        return True
    except onvif.exceptions.ONVIFError:
        logger.debug("Camera does not support PTZ Services")
        return False


def get_stream_configuration(camera):
    # Create the media service
    mycam = ONVIFCamera(camera['hostname'] , camera['port'], camera['username'], camera['password'], '/usr/local/lib/python3.6/site-packages/wsdl')
    #resp = mycam.devicemgmt.GetHostname()
    #logging.debug ('My camera`s hostname: {}'.format(str(resp.Name)))

    media_service = mycam.create_media_service()

    # Get target profile
    media_profiles = media_service.GetProfiles()

    obj = media_service.create_type('GetStreamUri')
    obj.StreamSetup = {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}}
    profiles = {}
    for profile in media_profiles:
        resolutions = {}
        logging.debug ("====================================================================")
        logging.debug ("Media Profile : {}".format(profile))

        obj.ProfileToken = profile.token
        stream_uris = media_service.GetStreamUri(obj)
        snapshot_uris = media_service.GetSnapshotUri({'ProfileToken':profile.token})

        logging.debug ("====================================================================")
        logging.debug ("Stream URIs for this profile : {}".format(stream_uris))
        logging.debug ("Snapshot URIs for this profile : {}".format(snapshot_uris))

        # Get video encoder configuration options
        options = media_service.GetVideoEncoderConfigurationOptions({'ProfileToken':profile.token})
        logging.debug ("====================================================================")
        logging.debug ("VideoEncoder Options: {}".format(options))

        stream_fps = 0
        try:
            # Get encoder FPS limit if any
            stream_fps = profile.VideoEncoderConfiguration.RateControl.FrameRateLimit
        except AttributeError:
            pass

        # Give preference to the latest codecs
        hres = []

        try:
            for resolution in options.H264.ResolutionsAvailable:
                logging.debug("Resolution supported: {}".format(resolution))
                try:
                    hres.append((resolution.Width, resolution.Height))
                except AttributeError:
                    logging.exception("No such attribute")
            resolutions['H264'] = hres
        except AttributeError:
            pass
        try:
            mres = []
            for resolution in options.MPEG4.ResolutionsAvailable:
                logging.debug("Resolution supported: {}".format(resolution))
                try:
                    mres.append((resolution.Width, resolution.Height))
                except AttributeError:
                    logging.exception("No such attribute")
            resolutions['MPEG4'] = mres
        except AttributeError:
            pass
        try:
            jres = []
            for resolution in options.JPEG.ResolutionsAvailable:
                logging.debug("Resolution supported: {}".format(resolution))
                try:
                    jres.append((resolution.Width, resolution.Height))
                except AttributeError:
                    logging.exception("No such attribute")
            resolutions['JPEG'] = jres
        except AttributeError:
            pass

        # Cameras are returning URLs without username/passwords embedded
        # Even when they do not support digest authentication, this is
        # workaround. Some cameras are putting auth info in the query portion of the
        # URL, try to leave those untouched, not expected to work for all
        # variations of cameras
        # TODO: Need to look into encoding username and passwords to handle
        surl = urlparse(stream_uris.Uri)
        authkeywords = ['username=', 'user=', 'password=']
        if (not surl.username or not surl.password) and (not any(x in surl.query for x in authkeywords)):
            logger.debug("URL parsed is {}".format(surl))
            netloc = "{}:{}@{}".format(camera['username'], camera['password'],surl.netloc)
            stream_uris.Uri = urlunparse((surl.scheme, netloc, surl.path, surl.params, surl.query, surl.fragment))
            logger.debug("URL with Auth is {}".format(stream_uris))


        profiles[profile.Name] = { 'stream_uris': stream_uris.Uri,
                                   'snapshot_uris': snapshot_uris.Uri,
                                   'stream_resolutions': resolutions,
                                   'stream_fps': stream_fps
                                  }

    logging.debug ("====================================================================")
    return profiles


def configwizard(config_file):
    if config_file is None:
        print ("Please specify a configuration file write to")
        return

    import os
    from getpass import getpass
    if os.path.isfile(config_file) is True:
        print("Configuration file: {} already exists ".format(config_file))
        print("Please specify a different file, or delete this file and rerun")
        return

    try:
        max_retries = 3
        max_frames_to_buffer = 16
        tomldict = {}
        cameras = discover_cameras()
        yesList = ['y', 'Y', 'Yes', 'YES', 'YEs', 'yes', 'yES', 'yEs', 'YE', 'Ye', 'ye', 'yE']
        common_username = None
        common_password = None
        camerastomonitor = []
        for camera in cameras:
            for i in range(max_retries):
                if common_username and common_password:
                    camera['username'] = common_username
                    camera['password'] = common_password
                    logger.debug("Set common username and password")

                addcamera = input ("Do you want add camera: {} y/n: ".format(camera['hostname']))
                if addcamera not in yesList:
                   break

                if not camera['username']:
                    camera['username'] = input ("Please enter the username for camera {}: ".format(camera['hostname']))
                if not camera['password']:
                    #camera['password'] = input ("Please enter the password for camera {}: ".format(camera['hostname']))
                    camera['password'] = getpass(prompt=f"Please enter the password for camera {camera['hostname']}: ")
                #print (f"using username:{camera['username']}, password: {camera['password']}")

                if not common_username or not common_password:
                    usecommon = input("Should I use these credentials for the rest of the discovered cameras? y/n:")
                    if usecommon in yesList:
                        common_username = camera['username']
                        common_password = camera['password']

                try:
                    camera['name'] = get_camera_name(camera)
                   #get_media_profile_configuration(camera)
                    profiles = get_stream_configuration(camera)
                    defaultprofile =  None
                    for profilename in profiles:
                        print ("Profile Name: {}".format(profilename))
                        if not defaultprofile:
                            defaultprofile = profilename
                        for encoding in profiles[profilename]['stream_resolutions']:
                            print ("  Encoding: {}".format(encoding))
                            print ("    Resolutions Available: {}\n".format(profiles[profilename]['stream_resolutions'][encoding]))
                    response = None
                    print ("Enter the name of the profile you want to use: ")
                    while response not in profiles.keys():
                        response = input("Enter to accept default '{}', or type in your selection and Enter: ".format(defaultprofile)) or defaultprofile
                        logger.debug("response is {}".format(response))
                        if response in profiles.keys():
                            #camera['profiles'] = profiles[response]
                            proc = Popen([STREAM_PLAYER, profiles[response]['stream_uris']],
                                         shell=False, stdout=DEVNULL,
                                         stderr=DEVNULL)
                            print ("Opening the video stream for verification")
                            time.sleep(10)
                            streamconfirm= input (f"Do you see the camera stream?  y/n: ")
                            if streamconfirm in yesList:
                                print ("Great!")
                                proc.kill()
                                cameraname = input(f"Enter to accept '{camera['name']}' as camera name, or type in another name and Enter: ") or camera['name']
                                camera['name'] = cameraname
                                rotationconfirm= input (f"Do you want the camera image to be rotated: y/n: ")
                                if rotationconfirm in yesList:
                                    rotationangle = int(input (f"Please enter the angle (0-360) want to rotate the image by: y/n: "))
                                    print (f"You entered {rotationangle}")
                                    camera['rotation'] = rotationangle
                            else:
                                proc.kill()
                                nostream = input (f"Do you want to pick a different profile? Selecting no will skip this camera:  y/n: ")
                                if nostream in yesList:
                                    response = None
                                else:
                                    # Skip this camera
                                    continue

                    camera['url'] = profiles[response]['stream_uris']
                    camera['fps'] = profiles[response]['stream_fps']
                    camera['snapshoturl'] = profiles[response]['snapshot_uris']

                    max_res_area = 0
                    max_res = {}
                    # Picking the highest resolution in any encoding as the max resolution
                    for encoding in profiles[profilename]['stream_resolutions']:
                        for resolution in profiles[response]['stream_resolutions'][encoding]:
                            if max_res_area < resolution[0]*resolution[1]:
                                max_res_area = resolution[0]*resolution[1]
                                max_res['Width'] = resolution[0]
                                max_res['Height'] = resolution[1]
                    camera['readbuffer'] = max_frames_to_buffer *  max_res['Width'] * max_res['Height']
                    camera['resolution'] = "{}x{}".format(max_res['Width'] , max_res['Height'])
                    camera['enableptz']  = is_ptz_supported(camera)

                    # Set some of the other entries
                    camera['subdir'] = './videos/' + camera['name']

                    # Update variables that are mandatory and not yet set
                    for key in CamAiConfig.CAMVARS:
                        if key not in camera:
                            camera[key] = CamAiConfig.CAMVARS[key]
                            logger.debug("Added {}: {}".format(key, camera[key]))

                    camerastomonitor.append(camera)
                    # No exceptions so break out of retry loop
                    break

                except onvif.exceptions.ONVIFError or zeep.exceptions.Fault as oe:
                    camera['username'] = None
                    camera['password'] = None
                    common_username = None
                    common_password = None
                    print("Error with supplied credentials, please retry")
                    print(f"Exception {oe}")
                    continue

            # confirm = input("Continue with rest of the cameras? y/n: ")
            # if confirm not in yesList:
            #     break

        # Handle the email and manager sections
        commondict = {}

        print("\n\n")
        print("Email alert setup")
        print("=================")
        print("In order to send email alerts to be accepted by your email provider, account information will be needed")
        print("To ensure encrypted delivery of alert images and videos, only TLS/SSL capable email servers are supported ")
        confirm = input("Do you want to setup email alerts? y/n: ")
        if confirm in yesList:
            defaultsmtpport = 465
            defaultsmtpserver = 'smtp.gmail.com'
            sender_email =  input("Please enter the email address of the sender:  ")
            print("In order to send email securely via TLS/SSL and through your email server, account information is needed")
            sender_login =  input("Please enter the username to login to the server to send email: e.g: username : ")
            #sender_secret=  input("Please enter the password for this account: ")
            sender_secret  = getpass(prompt=f"Please enter the password for this account: ")
            smtp_server =  input("Please enter the smtp server to send email through: e.g: smtp.gmail.com: ") or defaultsmtpserver
            smtp_server_port =  input("Please enter the smtp server port : e.g: 465: ") or defaultsmtpport

            commondict['email sender'] = {'sender_email': sender_email,
                              'sender_login': sender_login,
                              'smtp_server': smtp_server,
                              'smtp_server_port': smtp_server_port,
                              'login_required': True,
                              'use_SSL': True,
                              'sender_secret': sender_secret}
            addmore = 'y'
            email_recepients = []
            while addmore in yesList:
                addmore = 'n'
                name = input("Please enter the name of the person that will receive email alerts: ")
                email_address = input("Please enter the email address of the person that will receive email alerts: ")
                email_recepients.append({'name': name, 'email_address': email_address})
                addmore = input("Do you want to add more recepients? y/n: ")

            commondict['email recepient'] =  email_recepients
        else:
            #commondict['email sender'] = {}
            pass

        commondict['manager options'] = CamAiConfig.MANAGERVARS

        tomldict['camera'] = camerastomonitor
        tomltextcameras = toml.dumps(tomldict)
        tomltextcommon =  toml.dumps(commondict)
        #tomlfile = open('my.toml', 'w')
        tomlfile = open(config_file, 'w')
        tomlfile.write(tomltextcameras)
        tomlfile.write(tomltextcommon)
        tomlfile.close()

        logging.debug("Config file generated: {} {}".format(tomltextcameras, tomltextcommon))

    except KeyboardInterrupt:
        print("\nCanceling camera discovery wizard")

def test():
    camera1 = {'hostname': '192.168.1.10',
                    'port': 80,
                    'username': 'admin',
                    'password': 'password'
                    }
    #set_camera_time(camera1)
    #get_camera_time(camera1)
    #get_camera_services(camera1)
    #get_device_service_capabilities(camera1)
    #get_camera_capabilities(camera1)
    #get_snapshot_uri(camera1)
    #get_media_profile_configuration(camera1)
    #get_access_policy(camera1)
    configwizard()
