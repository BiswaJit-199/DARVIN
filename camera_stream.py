# camera_stream.py - Webcam initialization for Windows or Pi

import cv2

def get_camera():
	return cv2.VideoCapture(0)  # Use 0 for default webcam