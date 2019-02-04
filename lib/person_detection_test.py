
import cv2

import numpy as np
import subprocess



classes = ['background', 'aeroplane', 'aeroplane', 'aeroplane', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# premodel = cv2.dnn.readNetFromCaffe('models/MobileNetSSD_deploy.prototxt', 'models/MobileNetSSD_deploy.caffemodel')

class Person_Detect(object):
    def __init__(self):
        self.person_counter = 1 #initialized to 1 for first person
        self.file_count = 0
        self.detected_count = 0



    def get_frame(self, frame):
        premodel = cv2.dnn.readNetFromCaffe('lib/models/MobileNetSSD_deploy.prototxt',
                                            'lib/models/MobileNetSSD_deploy.caffemodel')

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        premodel.setInput(blob)
        detections = premodel.forward()


        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.55:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3: 7] * np.array([w, h, w, h])
                stX, stY, enX, enY = box.astype('int')
                person = frame.copy()[stY: enY, stX: enX]

                label = '{}: {:.2f}%'.format(classes[idx], confidence * 100.0)
                cv2.rectangle(frame, (stX, stY), (enX, enY), colors[idx], 2)
                y = stY - 15 if (stY - 15) > 15 else stY + 15
                cv2.putText(frame, label, (stX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
                if classes[idx] == 'person':
                    # check if the detected object is a person
                    # print('person found')
                    person = frame[stY: enY, stX: enX]

        return frame


if __name__ == '__main__':
    Camera1 = Person_Detect()
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        frame = Camera1.get_frame(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
