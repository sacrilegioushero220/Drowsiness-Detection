from flask import Flask, Response
import cv2
from detect import *
import numpy as np
import tensorflow as tf
# from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from playsound import playsound
from threading import Thread
import mediapipe as mp
from utils import *
Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate

    
import threading

import pyttsx3

app = Flask(__name__)
video = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier()
# face_cascade.load(cv2.samples.findFile("static/haarcascade_frontalface_alt.xml"))
   
def start_alarm(sound):
    """Play the alarm sound"""
    playsound('data/alarm.mp3')

@app.route('/')
def index():
    return "Default Message"

def iou(box1,box2):
    return intersection(box1,box2)/union(box1,box2)

def union(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)

def intersection(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)

yolo_classes = ['dangerous_dip', 'no_entry', 'no_parking', 'overtaking_prohibited', 'speelimit_80']


def process_output(output, img_width, img_height):
    output = output[0].astype(float)
    output = output.transpose()

    boxes = []
    for row in output:
        prob = row[4:].max()
        if prob < 0.5:
            continue
        class_id = row[4:].argmax()
        label = yolo_classes[class_id]
        xc, yc, w, h = row[:4]
        x1 = (xc - w/2) / 640 * img_width
        y1 = (yc - h/2) / 640 * img_height
        x2 = (xc + w/2) / 640 * img_width
        y2 = (yc + h/2) / 640 * img_height
        boxes.append([x1, y1, x2, y2, label, prob])

    boxes.sort(key=lambda x: x[5], reverse=True)
    result = []
    while len(boxes) > 0:
        result.append(boxes[0])
        boxes = [box for box in boxes if iou(box, boxes[0]) < 0.7]
    return result

def gen_traffic(video):
    interpreter = Interpreter(model_path='./best_float32.tflite')  # load TFLite model
    interpreter.allocate_tensors()  # allocate
    input_details = interpreter.get_input_details()  # inputs
    output_details = interpreter.get_output_details()  # outputs


    while True:
        ret, frame = video.read()  # read frame from webcam
        img = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)  # resize frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
        input = np.array(img, dtype=np.float32)
        input = np.expand_dims(input, axis=0)
        input = input/255.0
        interpreter.set_tensor(input_details[0]['index'], input)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        result = process_output(output_data, 640, 640)

        for box in result:
            x1, y1, x2, y2, class_id, score = box
            x1 = int(x1 * frame.shape[1] / img.shape[1])
            y1 = int(y1 * frame.shape[0] / img.shape[0])
            x2 = int(x2 * frame.shape[1] / img.shape[1])
            y2 = int(y2 * frame.shape[0] / img.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_id}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
    
        
        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')  # display frame

def gen(video):
    face_mesh = mp.solutions.face_mesh
    draw_utils = mp.solutions.drawing_utils
    landmark_style = draw_utils.DrawingSpec((0,255,0), thickness=1, circle_radius=1)
    connection_style = draw_utils.DrawingSpec((0,0,255), thickness=1, circle_radius=1)

    STATIC_IMAGE = False
    MAX_NO_FACES = 2
    DETECTION_CONFIDENCE = 0.6
    TRACKING_CONFIDENCE = 0.5

    COLOR_RED = (0,0,255)
    COLOR_BLUE = (255,0,0)
    COLOR_GREEN = (0,255,0)

    LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
        185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]

    RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
    LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]


    LEFT_EYE_TOP_BOTTOM = [386, 374]
    LEFT_EYE_LEFT_RIGHT = [263, 362]

    RIGHT_EYE_TOP_BOTTOM = [159, 145]
    RIGHT_EYE_LEFT_RIGHT = [133, 33]

    UPPER_LOWER_LIPS = [13, 14]
    LEFT_RIGHT_LIPS = [78, 308]


    FACE=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

    frame_count = 0

    face_model = face_mesh.FaceMesh(static_image_mode=STATIC_IMAGE,
                                    max_num_faces= MAX_NO_FACES,
                                    min_detection_confidence=DETECTION_CONFIDENCE,
                                    min_tracking_confidence=TRACKING_CONFIDENCE)

    # Load model
    interpreter = tf.lite.Interpreter(model_path='./model_drowsiness.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    count = 0
    alarm_on = False
    alarm_sound = "./data/alarm.mp3"
    status1 = ''
    status2 = ''

    speech = pyttsx3.init()
    while True:
        result, image = video.read()
        height, width = image.shape[:2]
        # height = frame.shape[0]
        if result: 
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            outputs = face_model.process(image_rgb)

            if outputs.multi_face_landmarks:

                
                landmarks = outputs.multi_face_landmarks[0].landmark # get the landmark of the first face
                
                draw_landmarks(image, outputs, LEFT_EYE_TOP_BOTTOM, COLOR_RED)
                draw_landmarks(image, outputs, LEFT_EYE_LEFT_RIGHT, COLOR_RED)

                ratio_left =  get_aspect_ratio(image, outputs, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)

                draw_landmarks(image, outputs, RIGHT_EYE_TOP_BOTTOM, COLOR_RED)
                draw_landmarks(image, outputs, RIGHT_EYE_LEFT_RIGHT, COLOR_RED)

                ratio_right =  get_aspect_ratio(image, outputs, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
                
                ratio = (ratio_left + ratio_right)/2.0
                
                ratio_lips =  get_aspect_ratio(image, outputs, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)

                rightEyeImg = getRightEye(image, landmarks) # get the right eye image
                leftEyeImg = getLeftEye(image, landmarks) # get the left eye image
                leftEyeImg = cv2.resize(leftEyeImg, (145, 145)) # resize the left eye image
                rightEyeImg = cv2.resize(rightEyeImg, (145, 145)) # resize the right eye image
                leftEyeImg = leftEyeImg.astype('float32') / 255.0 # normalize the left eye image
                rightEyeImg = rightEyeImg.astype('float32') / 255.0   # normalize the right eye image
                leftEyeImg = img_to_array(leftEyeImg) # convert the left eye image to array
                rightEyeImg = img_to_array(rightEyeImg) # convert the right eye image to array
                leftEyeImg = np.expand_dims(leftEyeImg, axis=0) # expand the dimension of the left eye image
                rightEyeImg = np.expand_dims(rightEyeImg, axis=0) # expand the dimension of the right eye image
                interpreter.set_tensor(input_details[0]['index'], leftEyeImg)
                interpreter.invoke()
                pred1 = interpreter.get_tensor(output_details[0]['index'])
                interpreter.set_tensor(input_details[0]['index'], rightEyeImg)
                interpreter.invoke()
                pred2 = interpreter.get_tensor(output_details[0]['index'])
                status1 = np.argmax(pred1) # get the status of the left eye if closed or not
                status2 = np.argmax(pred2) # get the status of the right eye if closed or not

                if ratio > 4:
                    frame_count +=1
                else:
                    frame_count = 0

                if (status1 == 2 and status2 == 2):
                    count+=1
                    # if eyes are closed for 10 consecutive frames, start the alarm
                    if count > 0 and frame_count > 3.5:
                        cv2.putText(image, "Drowsiness Alert!!! It Seems you are sleeping.. please wake up", (100, height-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        if not alarm_on:
                            alarm_on = True
                            # play the alarm sound in a new thread
                            t = Thread(target=start_alarm, args=(alarm_sound,))
                            t.daemon = True
                            t.start()

                if ratio_lips < 1.7:
                    message = 'Drowsy Warning: You looks tired.. please take rest'
                    p = threading.Thread(target=run_speech, args=(speech, message)) #create new instance if thread is dead
                    p.start()
                    
                else:
                    # cv2.putText(image, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                    count = 0
                    alarm_on = False
        result, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
    
        
        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_traffic')
def video_feed_traffic():
    global video
    return Response(gen_traffic(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)