import math
# Indices for top and bottom lips in Mediapipe
from utils.constants import TOP_LIP
from utils.constants import BOTTOM_LIP

class YawnDetector:
	def __init__(self, threshold=20, consecutive_frames=15):
		self.yawn_counter = 0
		self.yawn_threshold = threshold  # Mouth opening distance (pixels)
		self.consecutive_frames = consecutive_frames
		self.yawn_detected = False

	def get_lip_distance(self, landmarks):
		top = landmarks[TOP_LIP[0]]
		bottom = landmarks[BOTTOM_LIP[1]]
		distance = math.hypot(bottom[0] - top[0], bottom[1] - top[1])
		return distance

	def detect(self, landmarks):
		if not landmarks or len(landmarks) < 468:
			return False

		mouth_open = self.get_lip_distance(landmarks)

		if mouth_open > self.yawn_threshold:
			self.yawn_counter += 1
			if self.yawn_counter >= self.consecutive_frames:
				self.yawn_detected = True
				return True
		else:
			self.yawn_counter = 0
			self.yawn_detected = False

		return False
