import os
from subprocess import Popen
import numpy as np
import queue
import threading
from multiprocessing import Process
import logging
import datetime
import time
import cv2 as cv

from . import CamAiMessage, CamAiDetection, CamAiDetectionFaces, CamAiCameraWriter
#import CamAiMessage
#import CamAiDetection
#import CamAiDetectionFaces
#import CamAiCameraWriter

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


# Inputs: Camera_Name, Alert Timestamp, Known People List, Unknown People Count
#
def notifyVerbally(cameraname, alert_text, basedir="./"):
    from gtts import gTTS
    from io import BytesIO

    logger.warn(
        "Notifier: Issuing warning on camera {}".format(cameraname))

    # Language in which you want to convert
    language = 'en'

    # Passing the text and language to the engine,slow=False tells
    # the module that the converted audio should have a high speed
    myobj = gTTS(text=alert_text, lang=language, slow=False)

    # Saving the converted audio in a mp3 file named
    # welcome
    alarmfile = basedir + "alarm_" + cameraname + ".mp3"
    myobj.save(alarmfile)

    # Playing the converted file in the background and move on
    #os_play_cmd = 'play ' + '"' + alarmfile + '"'
    #os.system(os_play_cmd)
    proc = Popen(['play', alarmfile], shell=False)
    return proc


# email_sender is dict and email_recepients is an array of dicts
def notifyEmail(cameraname, email_sender, email_recepients, alert_text, file_attachment):
    # import email
    import smtplib
    import ssl
    from email import encoders
    from email.mime.base import MIMEBase
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    subject = f"Alarm on camera {cameraname}: {alert_text}"
    body = alert_text

    if ('smtp_server' not in email_sender):
        logger.warn(
            "Notifier: No SMTP Server configured, cannot send email notifications")
        return
    if 'smtp_server_port' not in email_sender:
        logger.warn(
            "Notifier: No SMTP Server port configured, cannot send email notifications")
        return
    if ('login_required' in email_sender) and (
            email_sender['login_required'] is True):
        if 'sender_login' not in email_sender:
            logger.warn(
                "Notifier: No sender login configured, cannot send email notifications")
            return
        if 'sender_secret' not in email_sender:
            logger.warn(
                "Notifier: No sender password configured, cannot send email notifications")
            return

    for recepient in email_recepients:
        if ('email_address' not in recepient):
            logger.warn(
                "Notifier: No sender email address configured, will not send email notifications to this recepient")
            next

        recepient_email = recepient['email_address']
        # Create a multipart message and set headers
        emailmessage = MIMEMultipart()
        emailmessage["From"] = email_sender['sender_email']  # sender_email
        emailmessage["To"] = recepient_email
        emailmessage["Subject"] = subject
        # emailmessage["Bcc"] = receiver_email  # Recommended for mass emails

        # Add body to email
        emailmessage.attach(MIMEText(body, "plain"))

        # Add the alarm image to the email message
        try:
            with open(file_attachment, 'rb') as attachment:
                # set attachment mime and file name, the image type is png
                mime = MIMEBase('image', 'png', filename=file_attachment)
                # add required header data:
                mime.add_header(
                    'Content-Disposition',
                    'attachment',
                    filename=file_attachment)
                mime.add_header('X-Attachment-Id', '0')
                mime.add_header('Content-ID', '<0>')
                # read attachment file content into the MIMEBase object
                mime.set_payload(attachment.read())
                # encode with base64
                encoders.encode_base64(mime)
                # add MIMEBase object to MIMEMultipart object
                emailmessage.attach(mime)
        except FileNotFoundError:
            logger.warning("Notifier: Missing attachment file")

        # Add attachment to message and convert message to string
        text = emailmessage.as_string()
        logger.debug("Notifier: Created email to send ")

        # Log in to server using secure context and send email
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(email_sender['smtp_server'], email_sender['smtp_server_port'],
                                  context=context) as server:
                logger.debug(
                    "Notifier: Logging in to server {}:{}".format(
                        email_sender['smtp_server'],
                        email_sender['smtp_server_port']))
                server.login(
                    email_sender['sender_login'],
                    email_sender['sender_secret'])
                logger.debug("Notifier: Trying to send mail to: {} logged in server {}:{}".format(
                    recepient_email, email_sender['smtp_server'], email_sender['smtp_server_port']))
                # Send the mail
                server.sendmail(
                    email_sender['sender_email'],
                    recepient_email, text)
            logger.debug(
                "Notifier: Sent mail to {} with attachment {}".format(
                    recepient_email, file_attachment))
        except TimeoutError:
            logger.warn(
                "Notifier: Timeout while trying to send email via {}:{}".format(
                    email_sender['smtp_server'], email_sender['smtp_server_port']))

    return


def create_person_alert_text(cameraname, timestamp, num_people_detected, faces_found, unknownfacescount):

    # At least one person is known to us
    people_string = ''
    stranger_string = ''
    numfaces = len(faces_found)
    if numfaces > 0:
        for index, person in enumerate(faces_found):
            if index == numfaces - 1:
                people_string = people_string + person
            elif index == numfaces - 2:
                people_string = people_string + person + ' and '
            else:
                people_string = people_string + person + ', '

        if numfaces > 1:
            term = ' were '
        else:
            term = ' was '

        people_string = people_string + term + 'seen on camera ' + cameraname

        if unknownfacescount == 1:
            stranger_string = " along with " + "another person"
        elif unknownfacescount > 1:
            stranger_string = " along with " + str(unknownfacescount) + " other people"
        else:
            stranger_string = ''
    # There were people, but none that we are able to recognize
    elif unknownfacescount > 0:
        if unknownfacescount == 1:
            stranger_string = "An unknown person was seen on camera " + cameraname
        elif unknownfacescount > 1:
            stranger_string = str(unknownfacescount) + " unknown people were seen on camera " + cameraname
        else:
            stranger_string = ''
    # We did not detect any faces, only people, ideally we'd have the people
    # count passed in
    else:
        if num_people_detected == 1:
            people_string = "A person was seen on camera " + cameraname
        elif num_people_detected > 1:
            people_string = "{} people were seen on camera {}".format(num_people_detected, cameraname)
        else:
            logger.warn("No people detected, yet alerting?")
            pass

    alertstring = ''
    alertstring += people_string
    alertstring += stranger_string
    alertstring +=  " at {}".format(timestamp)

    return alertstring

def create_vehicle_alert_text(cameraname, timestamp, num_vehicles_detected):
    vehicle_string = ''

    if num_vehicles_detected == 1:
        vehicle_string = "A car was seen on camera " + cameraname
    elif num_vehicles_detected > 1:
        vehicle_string = "{} cars were seen on camera {}".format(num_vehicles_detected, cameraname)
    else:
        logger.warn("No cars detected, yet alerting?")
        pass

    alertstring = ''
    alertstring += vehicle_string
    alertstring +=  " at {}".format(timestamp)

    return alertstring


class CamAiNotification (object):

    def __init__(
                self,
                config,
                notification_queues
                ):

        self.config = config
        self.notification_queues = notification_queues
        self.start_time = time.time()
        logger.debug("Notifier: Monitoring {} queues".format(len(notification_queues)))

    @property
    def email_sender(self):
        return self.config.get_email_sender()

    @property
    def email_recepients(self):
        return self.config.get_email_recepients()

    @property
    def basedir(self):
        return self.config.get_basedir()

    def run_as_process(self):
        manager_options = self.config.get_manager_options()

        if (manager_options['multiprocessing_notifier'] is True):
            return True
        else:
            return False

    def start(self):
        if self.run_as_process():
            self.notifier = Process(target=self._process_notifications,
                                    args=([]),
                                    name="Notifier")
            #logger.debug("Notifier: Starting as a Process")

        else:
            self.notifier = threading.Thread(
                target=self._process_notifications, args=(
                    []), name="Notifier")
            self.notifier.do_notify = True
            #logger.debug("Notifier: Starting as a Thread")

        # Update to real starting time
        self.start_time = time.time()
        self.notifier.start()

    def stop(self):
        # setting this flag only works when running as a thread
        self.notifier.do_notify = False
        logger.warn("Notifier: Stopping")

    def join(self, waittime=10):
        self.notifier.join(waittime)
        logger.warn("Notifier: Join")

    def _process_notifications(self):
        notification_wait = 2
        logger.debug("Notifier: processing {} queues".format(len(self.notification_queues)))

        #self.detectionfaces = CamAiDetectionFaces.CamAiDetectionFaces()
        self.detectionfaces = None

        # notifier = threading.currentThread()
        # Parent cannot set this if invoked as a process
        if (self.run_as_process() is True):
            do_notify = True

        # while notifier.do_notify is True :
        while do_notify is True:
            try:
                if (self.run_as_process() is False):
                    # TODO: In progress exit notifications only through queues
                    # do_notify = notifier.do_notify
                    pass

                for notifier_queue in self.notification_queues:
                    try:
                        #logger.warning("Notifier: queue get ")
                        message = notifier_queue.get(True, notification_wait)
                        logger.warning("Notifier: queue get after ")

                        if message.msgtype == CamAiMessage.CamAiMsgType.notification:
                            self.handle_notification(message)
                        elif message.msgtype == CamAiMessage.CamAiMsgType.quit:
                            logger.warn("Notifier: Got a quit message, stopping notifier")
                            do_notify = False
                            break
                        else:
                            logger.warn("Notifier: Unknown message type receieved!")
                            pass
                    except queue.Empty:
                        pass
                    except AttributeError as ae:
                        logger.warn("Notifier: AttributeError in notifier loop: \n{} ".format(ae))
            except KeyboardInterrupt:
                logger.warn("Notifier: Got a keyboard interrupt, exiting")
                break

        logger.warning("****************************************************")
        logger.warn("Notifier: Returning")
        logger.warning("****************************************************")
        return

    def handle_notification(self, message):

        if self.detectionfaces is None:
            self.detectionfaces = CamAiDetectionFaces.CamAiDetectionFaces()

        # microseconds are too noisy to keep around for  notifications
        message.msgdata['timestamp'] = message.msgdata['timestamp'] - \
                datetime.timedelta(microseconds=message.msgdata['timestamp'].microsecond)

        # Sampled and Batched Image Version
        logger.debug("Notifier: Got an imagelist of length {}".format(len(message.msgdata['image'])))

        matchesarray = message.msgdata['objects detected']

        # Detected a person
        if are_these_objects_in_matches(matchesarray, [CamAiDetection.Person_Index]):
            logger.debug("Notifier: Detected new people")
            self.handle_person_notification(message, self.detectionfaces)

         # Detected some new vehicle coming in
        if are_these_objects_in_matches(matchesarray, CamAiDetection.Vehicle_Indexes):
            logger.warning("Notifier: Detected new vehicles")
            self.handle_vehicle_notification(message)
            pass

    # TODO Need to do some coalescing of duplicating alerts to avoid excessive notifications
    # Notification is not explicitly aware of face recognition being on or off,
    # parent detection is minimizing face detection processing here, should be
    # more explicit. Notification is not per camera, so this configuration is
    # not directly available, need to rethink this, face detection should
    # really be a part of detector processes instead, which means having to
    # maybe have multiple types of detectors being active as two tensorflow
    # instances don't seem to like to operate in the same process
    def handle_person_notification(self, message, detectionfaces):
        fps = int(24/9) # TODO: Fix hardcoded sampling of 9
        images = message.msgdata['image']
        matchesarray = message.msgdata['objects detected']
        cameraname = message.msgdata['cameraname']
        timestamp = message.msgdata['timestamp']

        logger.debug("Notifier: len matchesarray is: {}".format(len(matchesarray)))
        # TODO: Need to do one pass face checking instead of one per image
        # If facedetection is set to off, matchesarray
        # should be zero, and thus cropped_images, and the
        # rest of the code a NOOP

        cropped_images = get_cropped_objects(images, matchesarray, CamAiDetection.Person_Index)
        num_people_detected = get_object_count(matchesarray, CamAiDetection.Person_Index)
        logger.debug("Notifier: len of cropped_images is {}, num people: {}".format(len(cropped_images), num_people_detected))

        bestimage = images[0]
        bestmatch = 1
        max_num_unmatched_faces = 0
        faces_found = {} # Dedup faces found
        for image, object_crops in zip(images, cropped_images):
            for object_crop in object_crops:
                # TODO: Use instancethreshold from configuration file instead
                # of hardcoded default of 0.6,
                numfaces, matches, unmatched = detectionfaces.find_face_matches(object_crop)
                logger.debug(f"Notifier: Found these matches in objects: {matches}")
                logger.debug(f"Notifier: Could not match {unmatched} of {numfaces} faces")
                # TODO: Really should have unknown people
                # tracking across frames to correctly
                # determine real unknown count
                if max_num_unmatched_faces < unmatched:
                    max_num_unmatched_faces = unmatched
                logger.debug("Notifier: matches found : {}".format(matches))
                for key in matches:
                    faces_found[matches[key][0]] = matches[key][1]
                    if bestmatch > matches[key][1]:
                        bestmatch = matches[key][1]
                        # This only finds best known faces, unlike check_for_faces which
                        # includes unknown faces in best selection
                        bestimage = image
        #logger.warning("Found these people: {}".format(faces_found))

        alarm_image_file = os.path.join(self.basedir, message.msgdata['cameraname'] + "_alarm_" + str(message.msgdata['timestamp']) + ".png")

        # Max compression to optimize for emailability
        # TODO: Might want to make the image size configurable
        compression_params = [cv.IMWRITE_PNG_COMPRESSION, 9]
        rc = cv.imwrite(alarm_image_file, bestimage, compression_params)

        alarm_video_file = os.path.join(self.basedir, message.msgdata['cameraname']+ "_alarm_"+ str(message.msgdata['timestamp'])+ ".mp4")
        CamAiCameraWriter.write_images_to_video(images, alarm_video_file, fps=fps)

        alert_text = create_person_alert_text(cameraname, timestamp, num_people_detected, faces_found, max_num_unmatched_faces)

        # Verbal notification in the background
        proc = notifyVerbally(cameraname, alert_text, basedir="./",)

        # TODO: Make it user configurable to send just the email with image or just video or both
        # Email notification with image
        notifyEmail(cameraname, self.email_sender, self.email_recepients, alert_text, alarm_image_file)

        # Email notification with video
        notifyEmail(cameraname, self.email_sender, self.email_recepients, alert_text, alarm_video_file)

        if proc.poll() is None:
            logger.warning("Notifier: Verbal notification still running at return time ")

    def handle_vehicle_notification(self, message):
        fps = int(24/9) # TODO: Fix hardcoded sampling of 9
        images = message.msgdata['image']
        matchesarray = message.msgdata['objects detected']
        cameraname = message.msgdata['cameraname']
        timestamp = message.msgdata['timestamp']

        bestimage = images[0]
        logger.debug("Notifier: len matchesarray is: {}".format(len(matchesarray)))

        # This is really only to draw rectangles around detected cars to debug
        # false detects
        cropped_images = get_cropped_objects(images, matchesarray, CamAiDetection.Car_Index)
        num_vehicles_detected = get_object_count(matchesarray, CamAiDetection.Car_Index)

        alarm_image_file = os.path.join(self.basedir, message.msgdata['cameraname'] + "_alarm_" + str(message.msgdata['timestamp']) + ".png")

        # Max compression to optimize for emailability
        # TODO: Might want to make the image size configurable
        compression_params = [cv.IMWRITE_PNG_COMPRESSION, 9]
        rc = cv.imwrite(alarm_image_file, bestimage, compression_params)

        alarm_video_file = os.path.join(self.basedir, message.msgdata['cameraname']+ "_alarm_"+ str(message.msgdata['timestamp'])+ ".mp4")
        CamAiCameraWriter.write_images_to_video(images, alarm_video_file, fps=fps)

        alert_text = create_vehicle_alert_text(cameraname, timestamp, num_vehicles_detected)

        # Verbal notification
        proc = notifyVerbally(cameraname, alert_text, basedir="./",)

        # TODO: Make it user configurable to send just the email with image or just video or both
        # Email notification with image
        notifyEmail(cameraname, self.email_sender, self.email_recepients, alert_text, alarm_image_file)

        # Email notification with video
        notifyEmail(cameraname, self.email_sender, self.email_recepients, alert_text, alarm_video_file)

        if proc.poll() is None:
            logger.warning("Notifier: Verbal notification still running at return time ")


# MatchesArray = [matches_dicts, ....}
# matches_dict =
# {
# 'object_index0': {'matched_records': [{'class': c, 'score': sc, 'roi': roi},... N]
#                   'found': True, # Should always be true as object_index entries that weren't found aren't included
#                   'best_score': best_score,
#                   'best_match_index': best_index
#                  }
# ....
# 'object_indexN': {'matched_records': [{'class': c, 'score': sc, 'roi':  roi},... N]
#                   'found': True, # Should always be true as object_index entries that weren't found aren't included
#                   'best_score': best_score,
#                   'best_match_index': best_index
#                  }
# }

# Returns true if there are one or more objects of the passed in types in
# matches
# ObjectIndex = array of objectindexes
# TODO: Refactor this whole demuxing part to be better organized
def are_these_objects_in_matches(matchesarray, object_indexes):
    for matches_dict in matchesarray:
        for object_index in object_indexes:
            if object_index in  matches_dict:
                return True

    return False


#def get_object_count(matchesarray, object_index=CamAiDetection.Person_Index):
def get_object_count(matchesarray, object_index):
    max_detected_count = 0
    object_class_name = CamAiDetection.get_class_name(object_index)

    for matches_dict in matchesarray:
        if (len(matches_dict) == 0) or (object_index not in matches_dict):
            continue
        logger.debug("Notifier: matches_dict is  {}, len: {}".format(matches_dict, len(matches_dict)))
        logger.debug("Notifier: matches_dict len: {}".format(len(matches_dict)))
        try:
            # Get the matched_records
            matched_records_for_object = matches_dict[object_index]['matched_records']
            #logger.warning("Notifier:  number of matches: {}".format(len(matched_records_for_object)))
            detected_count = 0

            for match in matched_records_for_object:
                logger.debug("Notifier: match for this object: {}\n".format(match))
                if match['class'] == object_class_name:
                    detected_count += 1

            if detected_count > max_detected_count:
                max_detected_count = detected_count
        except KeyError:
            logger.exception("Notifier: No matches in dict {}, for Index {}.".format(matches_dict, object_index))

    logger.debug("Notifier: object count is: {}".format(max_detected_count))
    return max_detected_count

# This will return an array of cropped objects per each input image.
# The returned images can be greater in number than input images if multiple
# objects are detected per image
# input images = [img1, img2, img3]
# output images = [ [img1crop1, img1crop2, img1crop3],
#                   [img2crop1, img2crop2],
#                   [img3crop1 ],
#                   [img4crop1, img4crop2 ],
#                   [],
#                   [img6crop1, img6crop2 ],
#def get_cropped_objects(images, matchesarray, object_index=CamAiDetection.Person_Index):
def get_cropped_objects(images, matchesarray, object_index):
    cropped_images = []

    for image, matches_dict in zip(images, matchesarray):
        if (len(matches_dict) == 0) or (object_index not in matches_dict):
            continue
        logger.debug("Notifier: matches_dict is  {}, len: {}".format(matches_dict, len(matches_dict)))
        logger.debug("Notifier: matches_dict len: {}".format(len(matches_dict)))
        try:
            # Get the matched_records
            #logger.warning("Notifier: matches_dict for this object is : {} ".format(matches_dict))
            matched_records_for_object = matches_dict[object_index]['matched_records']
            #logger.warning("\nNotifier: matches_dict for this object is : {} ".format(matches_dict))
            #logger.warning("\nNotifier: matched_records for this object: {}\n".format(matched_records_for_object))

            for match in matched_records_for_object:
                logger.debug("\nNotifier: match for this object: {}\n".format(match))
                confidence = False
                (h, w) = image.shape[:2]
                img_crops = []
                y1, x1, y2, x2 = match['roi']
                color = tuple(255 * np.random.rand(3))
                image = cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
                # Make the crop square so we can feed more of the pertinent image than
                # zero pads, We also expand the area by 21% to account
                # for bounding boxes being smaller than actual objects. This happens
                # quite often when objects are partially
                # occluded
                cy1 = int(y1*.9)
                cx1 = int(x1*.9)
                cy2 = int(y2*1.1)
                cx2 = int(x2*1.1)
                if (cy2-cy1) > (cx2-cx1):
                    cx2 = min(cx1+(cy2-cy1), w)
                elif (cx2-cx1) > (cy2-cy1):
                    cy2 = min(cy1+(cx2-cx1), h)

                objectcrop = image[cy1:cy2, cx1:cx2]
                img_crops.append(objectcrop)

                logger.debug("Notifier: Number of crops in this image is {}".format(len(img_crops)))
                cropped_images.append(img_crops)

        except KeyError:
            logger.exception("Notifier: No matches in dict {}, for Index {}.".format(matches_dict, object_index))

    logger.debug("Notifier: object crops being returned is: {}".format(len(cropped_images)))
    return cropped_images

