import logging
import toml
# Required for WS-Discovery (Multicast / UDP based)
import re
import sys
from urllib.parse import urlparse
from urllib.parse import urlunparse
from urllib.parse import urlencode

# Required for ONVIF queries to discovered cameras
import onvif
from onvif import ONVIFCamera
import zeep
import asyncio, sys
import time

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

DEBUG = False


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
    logging.debug ("Services : {}".format(services))
    for service in services:
        logging.debug ("Service: {} : ".format(service))

def get_device_service_capabilities(camera):
    mycam = ONVIFCamera(camera['hostname'] , camera['port'], camera['username'], camera['password'], '/usr/local/lib/python3.6/site-packages/wsdl')

    # Get List of Services
    devicemgmt_service = mycam.create_devicemgmt_service()
    servcapreq = devicemgmt_service.create_type('GetServiceCapabilities')
    service_caps = devicemgmt_service.GetServiceCapabilities(servcapreq)
    logging.debug ("Service Capabilities: {}".format(service_caps))
    for servicecap in service_caps:
        logging.debug ("Service capability: {} : ".format(servicecap))

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
    resp = mycam.devicemgmt.GetHostname()
    logging.debug ('My camera`s hostname: {}'.format(str(resp.Name)))
    dt = mycam.devicemgmt.GetSystemDateAndTime()
    tz = dt.TimeZone
    year = dt.UTCDateTime.Date.Year
    hour = dt.UTCDateTime.Time.Hour

    logging.debug ('My camera`s time: {}'.format(dt))

def get_camera_name(camera):
    mycam = ONVIFCamera(camera['hostname'] , camera['port'], camera['username'], camera['password'], '/usr/local/lib/python3.6/site-packages/wsdl')

    # Get Hostname
    resp = mycam.devicemgmt.GetHostname()
    logging.debug ('My camera`s hostname: {}'.format(str(resp.Name)))
    return resp.Name


XMAX = 1
XMIN = -1
YMAX = 1
YMIN = -1
ZMAX = 1
ZMIN = -1
moverequest = None
ptz = None
active = False
movestep = 500/1000

def increase_stepsize():
    global movestep
    movestep = movestep * 1.25

def decrease_stepsize():
    global movestep
    movestep = movestep / 1.25

def do_move(ptz, request):
    # Start continuous move
    global active
    global step
    if active:
        ptz.Stop({'ProfileToken': request.ProfileToken})
    active = True
    ptz.ContinuousMove(request)
    time.sleep(movestep)
    ptz.Stop({'ProfileToken': moverequest.ProfileToken})

def move_up(ptz, request):
    logging.debug ('move up...')
    #request.Velocity.PanTilt.x = 0
    #request.Velocity.PanTilt.y = YMAX
    request.Velocity = {'PanTilt': {'x': 0, 'y': YMAX}}
    do_move(ptz, request)

def move_down(ptz, request):
    logging.debug ('move down...')
    request.Velocity = {'PanTilt': {'x': 0, 'y': YMIN}}
    #request.Velocity.PanTilt.x = 0
    #request.Velocity.PanTilt.y = YMIN
    do_move(ptz, request)

def move_right(ptz, request):
    logging.debug ('move right...')
    request.Velocity = {'PanTilt': {'x': XMAX, 'y': 0}}
    #request.Velocity.PanTilt.x = XMAX
    #request.Velocity.PanTilt.y = 0
    do_move(ptz, request)

def move_left(ptz, request):
    logging.debug ('move left...')
    request.Velocity = {'PanTilt': {'x': XMIN, 'y': 0}}
    #request.Velocity.PanTilt.x = XMIN
    #request.Velocity.PanTilt.y = 0
    do_move(ptz, request)

def move_upleft(ptz, request):
    logging.debug ('move up left...')
    request.Velocity = {'PanTilt': {'x': XMIN, 'y': YMAX}}
    #request.Velocity.PanTilt.x = XMIN
    #request.Velocity.PanTilt.y = YMAX
    do_move(ptz, request)

def move_upright(ptz, request):
    logging.debug ('move up left...')
    request.Velocity = {'PanTilt': {'x': XMAX, 'y': YMAX}}
    #request.Velocity.PanTilt.x = XMAX
    #request.Velocity.PanTilt.y = YMAX
    do_move(ptz, request)

def move_downleft(ptz, request):
    logging.debug ('move down left...')
    request.Velocity = {'PanTilt': {'x': XMIN, 'y': YMIN}}
    #request.Velocity.PanTilt.x = XMIN
    #request.Velocity.PanTilt.y = YMIN
    do_move(ptz, request)

def move_downright(ptz, request):
    logging.debug ('move down left...')
    request.Velocity = {'PanTilt': {'x': XMAX, 'y': YMIN}}
    #request.Velocity.PanTilt.x = XMAX
    #request.Velocity.PanTilt.y = YMIN
    do_move(ptz, request)

def zoom_in(ptz, request):
    logging.debug ('zoom in ...')
    request.Velocity = {'Zoom': {'x': ZMAX}}
    do_move(ptz, request)

def zoom_out(ptz, request):
    logging.debug ('zoom out ...')
    request.Velocity = {'Zoom': {'x': ZMIN}}
    do_move(ptz, request)

def set_homeposition(ptz, request):
    # Start continuous move
    global active
    global step
    if active:
        ptz.Stop({'ProfileToken': request.ProfileToken})
    active = True
    ptz.SetHomePosition({'ProfileToken': request.ProfileToken})

def goto_homeposition(ptz, request):
    # Start continuous move
    global active
    global step
    if active:
        ptz.Stop({'ProfileToken': request.ProfileToken})
    active = True
    request.Velocity = {'PanTilt': {'x': XMAX, 'y': YMAX}}
    ptz.GotoHomePosition(request)

def get_presets(ptz, request):
    presets = ptz.GetPresets({'ProfileToken': moverequest.ProfileToken})
    logger.warning("Presets are {}".format(presets))

def set_preset(ptz, request):
    presetnum = input ("Please enter preset number: ")
    #presetname = input ("Please enter a name for this preset: ")
    presetrequest = ptz.create_type('SetPreset')
    presetrequest.ProfileToken = request.ProfileToken
    presetrequest.PresetToken = presetnum
    presetrequest.PresetName = None
    setresponse = ptz.SetPreset(presetrequest)
    logger.warning("SetPreset response {}".format(setresponse))

def goto_preset(ptz, request):
    presetnum = input ("Please enter preset number: ")
    # Start continuous move
    global active
    global step
    if active:
        ptz.Stop({'ProfileToken': request.ProfileToken})
    active = True

    presetrequest = ptz.create_type('GotoPreset')
    presetrequest.ProfileToken = request.ProfileToken
    presetrequest.PresetToken = presetnum
    presetrequest.Speed = {'PanTilt': {'x': XMAX, 'y': YMAX}}
    gotoresponse = ptz.GotoPreset(presetrequest)
    logger.warning("GotoPreset response {}".format(gotoresponse))

def setup_move(camera):
    mycam = ONVIFCamera(camera['hostname'] , camera['port'], camera['username'], camera['password'], '/usr/local/lib/python3.6/site-packages/wsdl')
    # Create media service object
    media = mycam.create_media_service()

    # Create ptz service object
    global ptz
    try:
        ptz = mycam.create_ptz_service()
    except onvif.exceptions.ONVIFError:
        logger.warning("Camera does not support PTZ Services")
        return False

    logging.debug ("Created ptz_service")

    # Get target profile
    media_profile = media.GetProfiles()[0]

    # TODO: If user selects a different profile to monitor, we
    # might want to use that, however unlikely that cameras would have
    # different PTZ drivers for each profile
    logging.debug ("Got profile for index 0, token is {} profile is {}".format(media_profile.token, media_profile))
    #logging.debug ("Got profile for index 0, token is {} ".format(media_profile.token))

    # Get PTZ configuration options for getting continuous move range
    request = ptz.create_type('GetConfigurationOptions')
    request.ConfigurationToken = media_profile.PTZConfiguration.token
    logging.debug ("ptz token is  {}".format(request.ConfigurationToken))
    ptz_configuration_options = ptz.GetConfigurationOptions(request)

    logging.debug ("Got configuration options for ptz {}".format(ptz_configuration_options))
    logging.debug ("Got configuration options for ptz token {}".format(media_profile.PTZConfiguration.token))

    global moverequest, stoprequest
    global XMAX, XMIN, YMAX, YMIN, ZMIN, ZMAX
    # Get range of pan and tilt
    # NOTE: X and Y are velocity vector
    XMAX = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Max
    XMIN = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Min
    YMAX = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Max
    YMIN = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Min
    ZMIN = ptz_configuration_options.Spaces.ContinuousZoomVelocitySpace[0].XRange.Min
    ZMAX = ptz_configuration_options.Spaces.ContinuousZoomVelocitySpace[0].XRange.Max
    logging.debug ("Done setting up global x,y max and min {} {} {} {} ".format(XMAX,XMIN,YMAX,YMIN))

    moverequest = ptz.create_type('ContinuousMove')
    logging.debug ("moverequest is {}".format(moverequest))
    #moverequest.Velocity = {'PanTilt': {'x': 1.0, 'y': 1.0}, 'Zoom': {'x': 1.0}}
    #moverequest.Velocity = {'PanTilt': {'x': 1.0, 'y': 1.0}, 'Zoom': {'x': 1.0}}
    moverequest.Velocity = {'PanTilt': {'x': 0.0, 'y': 0.0}, 'Zoom': {'x': 0.0}}
    moverequest.ProfileToken = media_profile.token
    movresponse = ptz.ContinuousMove(moverequest)

    if moverequest.Velocity is None:
        logging.debug ("Trying to get current status")
        getstatusrequest = ptz.create_type('GetStatus')
        getstatusrequest.ProfileToken = media_profile.token
        #getstatusrequest.ProfileToken = media_profile.PTZConfiguration.token
        status  = ptz.GetStatus(getstatusrequest)
        #status  = ptz.GetStatus({'ProfileToken': media_profile.token})
        logging.debug ("Got current status {}".format(status))
        moverequest.Velocity = status.position
        logging.debug ("Got current Velocity {}".format(moverequest.Velocity))

    # logging.debug ("Setting up stop request ")
    # stoprequest = ptz.create_type('Stop')
    # stoprequest.ProfileToken = media_profile.token
    # stoprequest.PanTilt = True
    # stoprequest.Zoom = True
    return True


def readin():
    """Reading from stdin and displaying menu"""
    global moverequest, ptz

    selection = sys.stdin.readline().strip("\n")
    lov=[ x for x in selection.split(" ") if x != ""]
    if lov:

        if lov[0].lower() in ["u","up"]:
            move_up(ptz,moverequest)
        elif lov[0].lower() in ["d","do","dow","down"]:
            move_down(ptz,moverequest)
        elif lov[0].lower() in ["l","le","lef","left"]:
            move_left(ptz,moverequest)
        elif lov[0].lower() in ["l","le","lef","left"]:
            move_left(ptz,moverequest)
        elif lov[0].lower() in ["r","ri","rig","righ","right"]:
            move_right(ptz,moverequest)
        elif lov[0].lower() in ["ul"]:
            move_upleft(ptz,moverequest)
        elif lov[0].lower() in ["ur"]:
            move_upright(ptz,moverequest)
        elif lov[0].lower() in ["dl"]:
            move_downleft(ptz,moverequest)
        elif lov[0].lower() in ["zi","zoom in","zin","zoomi"]:
            zoom_in(ptz,moverequest)
        elif lov[0].lower() in ["zo","zoom out","zout","zoomo"]:
            zoom_out(ptz,moverequest)
        elif lov[0].lower() in ["s","st","sto","stop"]:
            ptz.Stop({'ProfileToken': moverequest.ProfileToken})
            active = False
        elif lov[0].lower() in ["move faster","movf","mf","movfast"]:
            increase_stepsize()
        elif lov[0].lower() in ["move slower","movs","ms","movslow"]:
            decrease_stepsize()
        elif lov[0].lower() in ["set home","shome","sho","set home position"]:
            set_homeposition(ptz, moverequest)
        elif lov[0].lower() in ["go home","ghome","gho","goto home position"]:
            goto_homeposition(ptz, moverequest)
        elif lov[0].lower() in ["get presets","gpre"]:
            get_presets(ptz, moverequest)
        elif lov[0].lower() in ["set presets","spre"]:
            set_preset(ptz, moverequest)
        elif lov[0].lower() in ["goto preset","gopre"]:
            goto_preset(ptz, moverequest)
        elif lov[0].lower() in ["status","sta","sta","stat"]:
            status = ptz.GetStatus({'ProfileToken': moverequest.ProfileToken})
            print(status)
        elif lov[0].lower() in ["dr"]:
            move_downright(ptz,moverequest)
        elif lov[0].lower() in ["q","quit","qu"]:
            print ("Ending PTZ")
            sys.exit(0)
        elif lov[0].lower() in ["help", "h", "he"]:
            print("""Commands:
            'up','down','left','right', 'ul' (up left),
            'ur' (up right), 'dl' (down left), 'dr' (down right),
            'zi' (zoom in), 'zo' (zoom out),
            'mf' (move faster),'ms' (move slower),
            'set home (sho)', 'go home (gho),
            'get presets (gpre)', 'set preset (spre), 'goto preset (gopre)',
            'status' (status), 'stop', 'quit', 'help'""")
        else:
            print("What are you asking?\tI only know, 'up','down','left','right', 'ul' (up left), \n\t\t\t'ur' (up right), 'dl' (down left), 'dr' (down right), 'zi' (zoom in), 'zo' (zoom out), 'status' (status) and 'stop'")

    print("")
    print("Your command: ", end='',flush=True)


def ptzoperate(camera):
    logging.debug ("********************************************************************")
    logging.debug ("=================> Setting up for the move <========================")
    logging.debug ("********************************************************************")
    if setup_move(camera):
        print("Camera Ready for Control :")
        loop = asyncio.get_event_loop()
        try:
            loop.add_reader(sys.stdin,readin)
            print("Use Ctrl-C, or enter 'q' to quit")
            print("Your command: ", end='',flush=True)
            loop.run_forever()
        except:
            pass
        finally:
            loop.remove_reader(sys.stdin)


if __name__ == '__main__':
    camera1 = {'hostname': '192.168.1.10',
                    'port': 80,
                    'username': 'admin',
                    'password': '123456'
                    }
    camera2 = {'hostname': '192.168.1.16',
                    'port': 8000,
                    'username': 'admin',
                    'password': 'password'
                    }


    #get_camera_time(camera)
    #get_camera_services(camera1)
    #get_device_service_capabilities(camera1)
    #get_camera_capabilities(camera1)
    #get_snapshot_uri(camera1)
    #get_media_profile_configuration(camera2)
    #get_access_policy(camera1)
    ptzoperate(camera1)
