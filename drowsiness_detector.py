# drowsiness_detector.py - Main drowsiness logic with Mediapipe and logging

import mediapipe as mp
import cv2
import numpy as np
import time
from eye_tracker import calculate_ear
from utils.constants import EAR_THRESHOLD, CONSEC_FRAMES
from voice_alert import speak

class DrowsinessDetector:
	def __init__(self):
		self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
		self.counter = 0
		self.alert_on = False

	def process(self, frame):
		h, w = frame.shape[:2]
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		results = self.face_mesh.process(rgb)

		if results.multi_face_landmarks:
			mesh = results.multi_face_landmarks[0]
			landmarks = [(int(p.x * w), int(p.y * h)) for p in mesh.landmark]

			# Indices for both eyes (mediapipe)
			left = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
			right = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]

			# Compute average EAR
			ear = (calculate_ear(np.array(left)) + calculate_ear(np.array(right))) / 2.0

			if ear < EAR_THRESHOLD:
				self.counter += 1
				if self.counter == CONSEC_FRAMES:
					print(f"Drowsiness detected. EAR: {ear:.3f}")
					speak("You are getting drowsy. Please take a break.")
					self.alert_on = True
			else:
				if self.counter > 0:
					print(f"EAR back to normal: {ear:.3f}")
				self.counter = 0
				self.alert_on = False

			return frame, ear

		return frame, None