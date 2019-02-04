from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import numpy as np
import json
from matplotlib import pyplot as plt
from facial_features import get_face
from person_detection import detect_person

class PICore():

	def __init__(self):
		self.flag = False
		self.featureMatrix = []
		self.detect = Person_Detect() 
		
	def processFrame(self, frame):
		#Input: raw frame
		#output: an altered frame with a box drawn around the individual and a label applied above the box.
		
		#STEP-01: Take the raw frame detect if a person is in the frame, if they are then draw a box around them. If not then output raw frame.
		
		#STEP-02: Segment the image
		#STEP-03: gather color features
		
		#STEP-04: gather face feature
		#Step-05: Gather skeletal feature
		#STEP-06: Gather other features
		#STEP-07: Create feature matrix
		#STEP-08: Hash feature matrix to create a char string version of matrix
		#STEP-09: Create an alpha-numeric key for the feature matrix.
		#Step-10: Perform lookup in database for key.
		#STEP-11: Apply label to image if found or create a label if not found.
		#STEP-12: output altered frame
		return frame
		

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
			if confidence > 0.75:	# model test for confidence more than 0.75
				idx = int(detections[0, 0, i, 1])
				if classes[idx] == 'person':
					box = detections[0, 0, i, 3: 7] * np.array([w, h, w, h])
					stX, stY, enX, enY = box.astype('int')
					person = frame.copy()[stY: enY, stX: enX]
					cv2.rectangle(frame, (stX, stY), (enX, enY), (255, 0, 0), 2)
					person = frame[stY: enY, stX: enX]
		return frame, person	
	
	def get_color_dictionary():
		'''Gets the color dictionary used for
		   classifying a color.The function load the file
		   color_dictionary.txt and returns the color dictionary.
		'''

		with open('color_dictionary.txt') as color_data:
			colors = json.load(color_data)
		return colors

	def get_shirt_color(frame, colors):
		'''Gets the shirt color from a passed in frame.
			@returns a vector with a single string value for shirt color.
		'''
		_, segment, yb = get_face(frame) #for now detect a face to use as reference for cropping a frame
		average_color = [] #will hold the color to be returned in a vector
		body = frame[yb:,:] #numpy array slice to get the body frame
		color = ('b', 'g', 'r') #hold 
		bgr = [] #will be used to hold the max b, g, r value for a histogram for one frame

		#The for loop will process a color histogram for the image
		for i, col in enumerate(color): 
			histr = cv2.calcHist([body], [i], None, [256], [0, 256])
			bgr.append(np.argmax(histr)) #append max value of b, g, and r
	   
		average_color.append(colors[str(int(bgr[0]))][str(int(bgr[1]))][str(int(bgr[2]))]) #get the color that matches the b, g, r values from the color dictionary and append to the vector

		return average_color

	def get_skin_color(frame, colors):
		''' This function gets the skin color of a passed in frame.
			@returns a vector with a single string value for skin color.
		'''

		#The try catch statements are used since the program will crash if a face isn't detected.
		try:
			_, segment, yb = get_face(frame)
		except AttributeError:
			return "face not found"
		try:
			h, w = segment.shape[:2]
		except AttributeError:
			return "face not found"
		
		average_color = []
		cropped = segment[h//5:h - h//5, w//5:w - w//5] #static values to slice the frame and only get face. Scales decently with distance.
		clone = cropped.copy()
		color = ('b', 'g', 'r')
		bgr = []
		
		for i, col in enumerate(color):
			histr = cv2.calcHist([clone], [i], None, [256], [0, 256])
			bgr.append(np.argmax(histr))

		average_color.append(colors[str(int(bgr[0]))][str(int(bgr[1]))][str(int(bgr[2]))])

		return average_color
		
	def buildFeatureMatrix(self, feature):
		self.featureMatrix.append(feature)
		
	def hashFeatureMatrix(self):
		hash = ""
		return hash
		
	def createLabel(self):
		pass
		
	def sim(self, featureMatrix):
		pass