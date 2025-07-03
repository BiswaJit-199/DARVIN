# visualizer.py

import cv2
import mediapipe as mp

class Visualizer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_style = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        self.hand_style = self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)

    def draw_face_mesh(self, frame, face_results):
        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.face_style,
                    connection_drawing_spec=self.face_style
                )

    def draw_hand_landmarks(self, frame, hand_results):
        if hand_results.multi_hand_landmarks:
            for landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.hand_style,
                    connection_drawing_spec=self.hand_style
                )
