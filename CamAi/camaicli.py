#!/usr/bin/env python3
import sys
import os
import argparse

sys.path.append('./')

def monitor(config_file) :
    if os.path.isfile(config_file) is False:
        print(f"Configuration file {args.monitor_config_file} does not exist, exiting ")
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

def discover(config_file):
    from CamAi import CamAiDiscover

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

def run_cli():

# TODO: Command style of cli, decide between this or arg style below (test vs --test)
#     parser = argparse.ArgumentParser(
#             description='Command line utilities for CamAi package',
#             usage='''camaicli <command> [<args>]
#
# The most commonly used git commands are:
#    monitor Monitor cameras specified in the configuration file
#    discover Run camera discovery wizard to create a camera configuration in the
#    ptzcontrol Control camera with point to zoom (PTZ)
# ''')
#     parser.add_argument('command', help='Subcommand to run')
#     # parse_args defaults to [1:] for args, but you need to
#     # exclude the rest of the args too, or validation will fail
#     args = parser.parse_args(sys.argv[1:2])
#     try:
#        eval(args.command)()
#     except NameError:
#         print (f'{args.command} is an Unrecognized command')
#
#     return
#

    #logger.debug('ARGV      :{}'.format(sys.argv[1:]))
    parser = argparse.ArgumentParser(description='CamAi command line')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--monitor", action="store", dest="monitor_config_file", nargs='?', const="aicam.toml",
                        help='''Monitor cameras specified in the configuration file,
                       looks for configuration in aicam.toml if no file is specified''')
    group.add_argument("--discover", action="store", dest="discover_config_file", nargs='?', const="aicam.toml",
                        help='''Run camera discovery wizard to create a camera configuration in the
                       file specified, writes to aicam.toml if no file is specified''')
    group.add_argument("--ptzcontrol", action="store", dest="ptzcontrol", nargs='?', const="aicam.toml",
                        help='''Control camera with point to zoom (PTZ) ''')
    args = parser.parse_args(sys.argv[1:2])

    #print(f"args are {args}")

    if args.monitor_config_file:
        monitor(args.monitor_config_file)

    elif args.discover_config_file:
        discover(args.discover_config_file)

    elif args.ptzcontrol:
        ptzcontrol()


if __name__ == '__main__':
    run_cli()


