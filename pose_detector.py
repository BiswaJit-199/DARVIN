# pose_detector.py - Head pose and hand-near-face detection logic

import mediapipe as mp
import cv2
import numpy as np
import time
from utils.head_pose_utils import model_points
from voice_alert import speak
# from logger import logger
from utils.constants import ALERT_COOLDOWN
import math
from visualizer import Visualizer
from yawning import YawnDetector

class PoseDetector:
	def __init__(self, enable_visualization=True):
		self.yawn_detector = YawnDetector()
		self.mp_face = mp.solutions.face_mesh
		self.face_mesh = self.mp_face.FaceMesh(refine_landmarks=True)
		self.mp_hands = mp.solutions.hands
		self.hands = self.mp_hands.Hands()
		self.last_alert_time = 0
		self.head_turn_counter = 0
		self.enable_visualization = enable_visualization
		if self.enable_visualization:
			self.visualizer = Visualizer()

	def get_head_pose(self, landmarks, w, h):
		image_points = np.array([
			(landmarks[1][0], landmarks[1][1]),     # Nose tip
			(landmarks[152][0], landmarks[152][1]), # Chin
			(landmarks[33][0], landmarks[33][1]),   # Left eye corner
			(landmarks[263][0], landmarks[263][1]), # Right eye corner
			(landmarks[78][0], landmarks[78][1]),   # Left mouth
			(landmarks[308][0], landmarks[308][1])  # Right mouth
		], dtype="double")

		focal_length = w
		center = (w / 2, h / 2)
		camera_matrix = np.array([
			[focal_length, 0, center[0]],
			[0, focal_length, center[1]],
			[0, 0, 1]
		], dtype="double")

		dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
		success, rotation_vector, _ = cv2.solvePnP(
			model_points, image_points, camera_matrix, dist_coeffs
		)

		rmat, _ = cv2.Rodrigues(rotation_vector)
		angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
		pitch, yaw, roll = angles

		return pitch, yaw, roll

	def is_looking_away(self, yaw):
		# Yaw angle threshold for looking away (in degrees)
		return abs(yaw) > 35

	def is_hand_near_face(self, hand_landmarks, face_landmarks):
		face_x = face_landmarks[1][0]
		hand_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
		return abs(face_x / 640 - hand_x) < 0.2  # 20% of frame width

	def process(self, frame):
		h, w = frame.shape[:2]
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		results_face = self.face_mesh.process(rgb)
		results_hands = self.hands.process(rgb)

		head_turned = False
		phone_usage = False
		yaw = 0  # Initialize yaw
		
		# Visualization (Windows only, optional)
		if self.enable_visualization:
			self.visualizer.draw_face_mesh(frame, results_face)
			self.visualizer.draw_hand_landmarks(frame, results_hands)

		if results_face.multi_face_landmarks:
			mesh = results_face.multi_face_landmarks[0]
			landmarks = [(int(p.x * w), int(p.y * h)) for p in mesh.landmark]

			pitch, yaw, roll = self.get_head_pose(landmarks, w, h)

			if self.is_looking_away(yaw):
				self.head_turn_counter += 1
				if self.head_turn_counter >= 15:  # ~? second
					head_turned = True
					if time.time() - self.last_alert_time > ALERT_COOLDOWN:
						speak("Please focus on the road.")
						print(f"Head turned. Yaw: {yaw:.2f}")
						self.last_alert_time = time.time()
			else:
				self.head_turn_counter = 0

			# Phone detection only if face landmarks found
			if results_hands.multi_hand_landmarks:
				for hand_landmarks in results_hands.multi_hand_landmarks:
					if self.is_hand_near_face(hand_landmarks, landmarks):
						phone_usage = True
						if time.time() - self.last_alert_time > ALERT_COOLDOWN:
							speak("Avoid using phone while driving.")
							print("Phone usage detected.")
							self.last_alert_time = time.time()
			
			# Yawn detection
			if self.yawn_detector.detect(landmarks):
				if time.time() - self.last_alert_time > ALERT_COOLDOWN:
					speak("You seem tired. Please take a break.")
					print("Yawning detected!")
					self.last_alert_time = time.time()

		return head_turned, phone_usage