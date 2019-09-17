#!/usr/bin/env python3
import sys
import os
import argparse

sys.path.append('./')

usage='''camaicli <command> [<args>]

 CamAi commands supported in this version are:
    help - Print this message
    monitor - Monitor cameras specified in the configuration file
    discover - Run camera discovery wizard to create a camera configuration in the
    ptzcontrol - Control camera with point to zoom (PTZ)
'''

#def monitor(config_file) :
def monitor() :
    myparser = argparse.ArgumentParser(description='Monitor Cameras')
    #myparser.add_argument("--config", action="store", dest="config_file", required=True,
    #                    help='''Configuration File''')
    myparser.add_argument("--config", action="store", dest="config_file", default='aicam.toml',
                        help='''Configuration File''')
    args = myparser.parse_args(sys.argv[2:])
    config_file = args.config_file

    if os.path.isfile(config_file) is False:
        print(f"Configuration file {config_file} does not exist, exiting ")
        return

    from CamAi import CamAiManager

    #print("Hello world")
    #os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "video_codec;h264_cuvid"
    #os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "video_codec;hevc_cuvid"
    #os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "video_codec;h264"
    #os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "rtsp_transport;tcp"
    #os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "video_codec;hevc_cuvid|rtsp_transport;tcp"
    #os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "video_codec;h264_cuvid|rtsp_transport;tcp"
    #os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "video_codec;hevc"
    #os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "video_codec;hevc|rtsp_transport;tcp"
    #os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "rtsp_transport;tcp"
    try:
        print("What codec am I using", os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'])
    except BaseException:
        print("No codec option set, using OS default")

    CamAiManager.start_cameras(config_file)

    print("Bye world")

#def discover(config_file):
def discover():
    from CamAi import CamAiDiscover

    myparser = argparse.ArgumentParser(description='Discover Cameras')
    #myparser.add_argument("--config", action="store", dest="config_file", required=True,
    #                    help='''Configuration File''')
    myparser.add_argument("--config", action="store", dest="config_file", default='aicam.toml',
                        help='''Configuration File''')
    args = myparser.parse_args(sys.argv[2:])
    config_file = args.config_file

    if os.path.isfile(config_file) is True:
        print("Configuration file: {} already exists ".format(config_file))
        print("Please specify a different file, or delete this file and rerun")
        return

    CamAiDiscover.configwizard(config_file)

def ptzcontrol():
    from CamAi import CamAiPtz

    myparser = argparse.ArgumentParser(description='Point and Zoom controls')
    myparser.add_argument("--ip", action="store", dest="camera_ip", required=True,
                        help='''IP Address of the Camera''')
    myparser.add_argument("--port", action="store", dest="camera_port", default='80',
                        help='''Port on which the Camera listens to''')
    myparser.add_argument("--username", action="store", dest="camera_username", default="admin",
                        help='''Username for the camera ''')
    myparser.add_argument("--password", action="store", dest="camera_passwd", default="123456",
                        help='''Password for the camera''')
    args = myparser.parse_args(sys.argv[2:])

    camera = {'hostname': args.camera_ip,
              'port': args.camera_port,
              'username': args.camera_username,
              'password': args.camera_passwd }

    print(f"camera to ptz control is {camera}")
    CamAiPtz.ptzoperate(camera)

def help():
    print (usage)

def run_cli():
    Supported_Commands = ['monitor', 'discover', 'ptzcontrol', 'reset', 'help']
# TODO: Command style of cli, decide between this or arg style below (test vs --test)
    parser = argparse.ArgumentParser(
            description='Command line utilities for CamAi package',
            usage=usage)
    parser.add_argument('command', help='Subcommand to run')
    # parse_args defaults to [1:] for args, but you need to
    # exclude the rest of the args too, or validation will fail
    args = parser.parse_args(sys.argv[1:2])
    if args.command in Supported_Commands:
        try:
           eval(args.command)()
        except NameError:
            print (f'{args.command} is not yet implemented')
    else:
        print (f"{args.command} is not a supported command")

    return


if __name__ == '__main__':
    run_cli()


