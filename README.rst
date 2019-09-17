=====
CamAI
=====
CamAi takes an AI first approach to monitoring and description of a video stream. Traditional open source packages like ZoneMinder and others largely rely on motion detection and handcrafted "zones" to try to reduce false detections. Modern Cameras support much higher resolutions (FHD, 4MP, 5MP, 4K etc) and this requires consistently low latency handling of the video streams, which the older systems seem to be having issues coping with. 

CamAi tries to take advantange of the 'many cpu cores' and is heavily threaded to ensure reliable high resolution video stream processing of 'many' sources. It also uses more modern approaches like Instance Segmentation, Face Detection and Face Recognition to reduce the false alarm rates. While this requires a GPU/s to get reasonable performance, inference engines bundled with traditional cpu cores are the future. It takes a data science approach to enable learning over time what might be normal or otherwise and get closer to a human in taking actions. Some of these features are available in the "cloud" cameras like Ring :sup:`tm`;, Nest :sup:`tm`;, which requires the video streams to be sent to 'their' cloud, and come with many involuntary 'sharing' features. CamAi can do this all within the network, with or without any external connectivity for the camera streams. 

It is certainly possible to try to 'graft' these features to existing systems. However it would have resulted in a mish mash of languages and technologies (perl/c++/windows only) or only partially open source. CamAI aims to have a more modern, maintainable and extensible code base that is fully be open source, and cross platform (over time) and modern, without having to jettison control and visibility to 'cloud' cameras.


============
Installation
============

Requirements
------------

The development has been primarily on a ubuntu 18.04 base. 

Install Ubuntu Packages in ubuntu_deps.txt.
    bash ./ubuntu-deps.txt

Install Nvidia CUDA for tensorflow acceleration
    Cuda 10.0(not 10.1), CuDNN 7.5 and Tensorflow 1.13/1.14 was used for testing
    Any compatible combination after these versions should work.

    It's possible to run tensorflow without CUDA, but it will most likely be impractically slow. 

Install Pip3 packages in requirements.txt
    Optional but Recommended: Create a virtual env for camai
        mkdir camai
        # If you do not have virtualenv already, or want to use a more current version         
        pip3 install virtualenv --user 
        virtualenv -p python3.7 venv37

    Unpack CamAi
        tar xvfz - camai.tgz (from tar bundle)
        pip3 install ./camai.whl (from a local wheel, pypi has size limits) 


    Install required python packages
        pip3 install -r ./CamAi/requirements.txt

Quick Run
    Discover onvif compatible cameras on your network and generate a configuration file
    
    ./CamAi/camaicli.py discover 
    or if installed from a wheel
    python3.7 ~/venv37/lib/python3.7/site-packages/CamAi/camaicli.py discover 

    Start monitoring with 
    ./CamAi/camaicli.py monitor 
     or if installed from a wheel
    python3.7 ~/venv37/lib/python3.7/site-packages/CamAi/camaicli.py monitor

    This should start logging videos to storage directories you specified in the config file.
    You cannot open the video being currently logged till it's rotated. 
    Default rotation is 30 minutes at hourly boundaries, so you should be able to open it after next hours starts. Alert images and snippet videos are generated in realtime and are located at the base directory for viewing.
