from imutils.video import VideoStream
import numpy as np
import imutils
import cv2


def detect_person(frame):
    '''
    this functions test if there is any person in the supplied video frame.
    If there is any person in the frame, it also crops the region where the person is
    and returns that region.
    :param frame:
    :return:
    '''
    classes = ['background', 'aeroplane', 'aeroplane', 'aeroplane', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']
    premodel = cv2.dnn.readNetFromCaffe('../models/MobileNetSSD_deploy.prototxt',
                                        '../models/MobileNetSSD_deploy.caffemodel')

    frame = imutils.resize(frame, width=300, height=300)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    premodel.setInput(blob)
    detections = premodel.forward() # using caffe model of mobilessdnet
    person = None

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.75:   # model test for confidence more than 0.75
            idx = int(detections[0, 0, i, 1])
            if classes[idx] == 'person':
                box = detections[0, 0, i, 3: 7] * np.array([w, h, w, h])
                stX, stY, enX, enY = box.astype('int')
                person = frame.copy()[stY: enY, stX: enX]
                cv2.rectangle(frame, (stX, stY), (enX, enY), (255, 0, 0), 2)
                person = frame[stY: enY, stX: enX]

    return frame, person

if __name__ == '__main__':
    vid = VideoStream(0).start()

    while (True):
        frame = vid.read()
        marked_frame, person = detect_person(frame.copy())
        cv2.imshow('person', marked_frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
