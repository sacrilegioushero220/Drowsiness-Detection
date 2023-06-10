import tensorflow as tf
Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
import cv2
import numpy as np



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



import cv2
import numpy as np
import tensorflow as tf

cap = cv2.VideoCapture(0)  # capture frames from webcam

# Load the TFLite model and allocate tensors
interpreter = Interpreter(model_path='best_float32.tflite')  # load TFLite model
interpreter.allocate_tensors()  # allocate
input_details = interpreter.get_input_details()  # inputs
output_details = interpreter.get_output_details()  # outputs


while True:
    ret, frame = cap.read()  # read frame from webcam
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

    cv2.imshow('frame', frame)  # display frame
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to exit
        break

cap.release()  # release webcam
cv2.destroyAllWindows()  # close all windows