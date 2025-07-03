# main.py - Driver distraction monitor main loop

import cv2
from camera_stream import get_camera
from drowsiness_detector import DrowsinessDetector
from pose_detector import PoseDetector
from voice_alert import speak

def run():
	cap = get_camera()
	detector = DrowsinessDetector()
	pose_detector = PoseDetector(enable_visualization=False)

	speak("Hello Boss, I am DARVIN.")
	speak("System started, Monitoring driver.")
	
	while True:
		ret, frame = cap.read()
		if not ret:
			speak("Failed to capture frame.")
			break

		frame = cv2.flip(frame, 1)
		output, ear = detector.process(frame)

		if ear:
			cv2.putText(output, f'EAR: {ear:.2f}', (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

		head_turned, phone_usage = pose_detector.process(frame)
		if head_turned:
			cv2.putText(output, "WARNING: Not looking forward!", (10, 60),
                		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		if phone_usage:
			cv2.putText(output, "WARNING: Phone detected!", (10, 90),
                		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.imshow("Driver Monitor", output)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			speak("System shutdown requested.")
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	run()
