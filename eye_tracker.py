# eye_tracker.py - EAR calculation logic
from numpy import linalg

def calculate_ear(eye):
	#Compute vertical and horizontal distances
	A = linalg.norm(eye[1] - eye[5])
	B = linalg.norm(eye[2] - eye[4])
	C = linalg.norm(eye[0] - eye[3])

	ear = (A + B) / (2.0 * C)
	return round(ear, 3)