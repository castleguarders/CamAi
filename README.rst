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

    It's possible to run tensorflow without CUDA, but it will most likely be impractically slow. If you use CamAi in record only mode, a GPU will not be essential.

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

    If you want to specify a configuration file in a different location you can add --config config.toml to the above
    python3.7 ~/venv37/lib/python3.7/site-packages/CamAi/camaicli.py monitor

    This should start logging videos to storage directories you specified in the config file.
    You cannot open the video being currently logged till it's rotated. Default rotation is 30 minutes at hourly boundaries, so you should be able to open it after next hours starts. Alert images and snippet videos are generated in realtime and are located at the base directory for viewing.


==============
Configuration
==============
Below is an example configuration file that the camcli.py --discover option generates. The configuration file can be hand crafted. There are three main sections.

**Cameras**
    There has to be one or more camera sections. This section contains camera specific configuration and policy information. The camera section consists of a main section and a sub section. The sub section is used to specify one or more objects of interest for this camera and the object specific settings.

**Email Section**
    This contains one Email Sender and zero or more Email Recepient sections

**Manager Section**
    This section has CamAi runtime defaults. If the user doesn't specify a per camera directory, the manager sections basedir variable is used as the parent for each camera.::

      [[camera]]
      hostname = "192.168.2.10"   <--- Largely ceremonial (Not mandatory)
      port = 8000                 <--- Onvif port  (Not mandatory)
      username = "admin"          <--- Not used yet, earlier versions used this to construct the URL
      password = "blahblah"       <--- Not used yet, earlier versions used this to construct the URL
      name = "Garage Walkway"     <--- Make this human consumable as it shows up in alerts/logs etc (Required)
      url = "rtsp://admin:blahblah@192.168.1.10:554/h264Preview_01_main" (Required)
      fps = 22                    <--- Gets used, use a sane number (Recommended)
      snapshoturl = "http://192.168.1.10:80/onvifsnapshot/snapshot_channel01.jpg" <--- (Not used yet)
      readbuffer = 78643200       <--- 16 x resolution is generally recommended (Required)
      resolution = "2560x1920"    <--- Not mandatory
      enableptz = false           <--- Not used yet
      subdir = "./videos/Garage Walkway"  <--- Recommended, manager sections basedir + name is used otherwise
      mode = "detect_and_record_timelapse" <--- Required
      rotation = 0                <--- Required if image is not oriented correctly
      watchdogenable = true       <--- Only set to false if camera is actually just a video file 
      watchdogtimer = 60          <--- Default is good
      annotation = false          <--- Leave to false other than when image masking figuring out false detects, intensive overhead
      showlivevideo = false       <--- Not used yet, older versions us
      facedetection = false       <--- Hmm, this needs to deprecated as 'instancedetection' under objects of interest is the right
      maxreadqueue = 32           <--- Tweak per your memory availability, memory consumed is this times resolution
      detectionrate = 25          <--- Detection frequency, these many frames are skipped between detections
      detecttrackseconds = 30     <--- Automatically track this many seconds if object of interest is detected
      deletepolicyenabled = true  <--- Enable automatic storage management
      deletethreshold = 60        <--- Start deleting when overall disk usage reaches this percentage
      deleteeventsthreshold = 80  <--- Start deleting videos with events when overall disk usage reaches this percentage
      deleteafterdays = 7         <--- Only delete after this many days
      log_object_timeseries = true <--- Leave at true, used for instance detection and reducing redundant notifications
      
      [camera.objects_of_interest.person] <--- Sub section for an object of interest, in this case a 'person' object
      detection_threshold = 0.98          <--- How confident the model should be that this object was detected
      instance_match_threshold = 75       <--- How confident the model should be that this is the known person based on face recognition
      instance_watch_timerange_start = 8  <--- Suppress alerts if the same object is seen within the last 8 minutes
      instance_watch_timerange_end = 2    <--- Works with above, when to start looking if the object was last seen, default is 2 mins ago
      instance_watch_continuous_mode = false  <--- The object has to be continually present within the watch timerange otherwise 
                                                   alerts will be triggered 
      instancedetection = false           <--- Face recognition if it's a person object, (licenses for vehicles will also use this)
      notify_startup_wait = 5             <--- How long to wait after startup before notifying, otherwise it will report existing objects

      [camera.objects_of_interest.car]
      detection_threshold = 0.85
      instance_match_threshold = 65
      instance_watch_timerange_start = 6
      instance_watch_timerange_end = 2
      instance_watch_continuous_mode = false
      instancedetection = false
      notify_startup_wait = 5

      [camera.objects_of_interest.motorcycle]
      detection_threshold = 0.98
      instance_match_threshold = 75
      instance_watch_timerange_start = 8
      instance_watch_timerange_end = 2
      instance_watch_continuous_mode = false
      instancedetection = false
      notify_startup_wait = 5

      [camera.objects_of_interest.truck]
      detection_threshold = 0.85
      instance_match_threshold = 65
      instance_watch_timerange_start = 6
      instance_watch_timerange_end = 2
      instance_watch_continuous_mode = false
      instancedetection = false
      notify_startup_wait = 5

      [camera.objects_of_interest.bus]
      detection_threshold = 0.98
      instance_match_threshold = 75
      instance_watch_timerange_start = 8
      instance_watch_timerange_end = 2
      instance_watch_continuous_mode = false
      instancedetection = false
      notify_startup_wait = 5

      [camera.objects_of_interest.bicycle]
      detection_threshold = 0.98
      instance_match_threshold = 75
      instance_watch_timerange_start = 8
      instance_watch_timerange_end = 2
      instance_watch_continuous_mode = false
      instancedetection = false
      notify_startup_wait = 5
      
      [["email recepient"]]  <--- Can have any number of recipients
      name = "First Last"   <--- Recipient Name
      email_address = "myemail@example.com"   <--- Recipient Email
      
      ["email sender"]
      sender_email = "firstlast@example.com"   <--- Email address from which the alerts are sent
      sender_login = "firstlast@example.com"   <--- Login to use to authenticate with the smtp server
      smtp_server = "mail.example.com"         <--- Your email providers SMTP
      smtp_server_port = 465                   <--- SSL SMTP Server Port
      login_required = true                    <--- Only authenticated SMTP is supported 
      use_SSL = true                           <--- Only SSL SMTP is supported 
      sender_secret = "blahblahpassword"       <--- The password
      
      ["manager options"]
      basedir = "./videos/"                    <--- Default storage location if not overriden on a per camera basis
      numdetectors = 1                         <--- Number of detector processes, if more than 1, multiprocessing_detector 
                                                    and multiprocessing_observer has to be true
      pipelineresize = true                    <--- Image resizing for detection, doing it per camera (vs in detector) reduces latency 
      singledetectorqueue = true               <--- Use a single queue for all cameras vs one queue for all cameras, single is cheaper
      defaultmaxqueue = 32                     <--- Default queue size, per camera override exists
      multiprocessing_observer = false         <--- Leave these settings as is for now
      multiprocessing_detector = false
      multiprocessing_reader = false
      multiprocessing_writer = false
      multiprocessing_notifier = true
      multiprocessing_viewer = false
      multiprocessing_trainer = true

**Choosing the Camera Operating Mode**
        CamAi supports four operating modes. The mode is configurable on a per camera basis. The default mode generated by the 

        *detect_and_record_timelapse* 
                Detection is on for the camera, and frames are recorded in a timelapse fashion at sample interval specified by 'detectionrate'. If any objects of interest are observed, the recording becomes continuous the number of seconds specified by 'detecttrackseconds' for that camera. If facedetection/instancedetection is enabled, the frames are further processed to scan for known instances before generating alerts. 

        *detect_and_record_everything*
                This mode works similar to detect_and_record_timelapse, except that all frames are recorded whether or not an object of interest is detected or otherwise.

        *detect_only*
                This mode works similar to detect_and_record_timelapse, except that no video is recorded (other than short alert clips). Alerting etc. works exactly the same.

        *record_only*
                This mode turns off any detection and records every frame.



**Setting up Face Recognition**
        Create a known faces directory. Follow the sample directory heirarchy.
        Put this under camai directory for now. Pictures in standard 
        formats like jpg/gif/png/bmp should work. Each picture should only have 
        the face of the person whose name is the parent directory.::

         known --->(Parent Directory)
               people --->(People Directory)
               │
               ├── barack obama  ------> (Directory is named after the person)
               │   └── barack.jpg 
               │   └── vacation1.jpg 
               │   └── dude3.jpg 
               └── Da Wife
               │   └── Wife1.png
               │   └── Spouse2.png
               │   └── IMG2019_3838383.JPEG 
               │   └── BossLady.png
               └── Da Hubby
               │   └── dameek1.png
               │   └── papertiger.png
               └── Da Son
                   └── slacker.jpg
                   └── gamerboy.jpg
                   └── prodigalreturns.jpg

**Alerts**
        The following types of alerts are supported.

        *Verbal*
                When a new object of interest is detected, it is announced locally on the computer thats running camai. 
.. image:: docs/Alerts_Storage.png
   :width: 50
   :height: 150
   :alt: Verbal Warning 


        *Email*
                When a new object of interest is detected, emails alerts are sent to the recipients specified in the configuration file. Each object of interest results in two emails, the first one with a single image and a follow up one with a short video clip. This is so that email size limits do not cause alerts to be missed.

.. image:: docs/Email_Alert_with_Image.png
   :width: 50
   :height: 150
   :alt: Email Alert with Image

.. image:: docs/Email_Alert_with_Video.png
   :width: 50
   :height: 150
   :alt: Email Alert with Video 



        *Stored*
                Every alerts image and video clipping is stored by default in the 'basedir' specified in the manager section. These alert files are not automatically pruned or managed unlike the video recordings.

.. image:: docs/Alerts_Storage.png
   :width: 50
   :height: 150
   :alt: Storage Alert Image 
                

