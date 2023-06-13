import cv2
import mediapipe as mp
import numpy as np
from src.analyzer import Analyzer


class PoseAnalysis(Analyzer):

    def __init__(self, video_path: str):
        super().__init__(video_path)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose_states = []

    def analyse(self, paddle_results: dict) -> list:
        print("Analysing pose...")
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret or frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                break

            # Process every 5 frames
            if self.current_frame_number % 5 == 0:
                pose_result = self.pose_analysis(frame)
                self.pose_states.append(
                    {"frame": self.current_frame_number, "stance": pose_result})

            self.current_frame_number += 1

        results = self.evaluate_results(paddle_results)
        return results

    @staticmethod
    def calculate_angle(a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
            np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def pose_analysis(self, frame):
        with self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as pose:
            stance = "None"

            frame.flags.writeable = True
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                elbow_left = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                shoulder_left = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip_left = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                wrist_left = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                elbow_right = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                shoulder_right = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip_right = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                # Calculate angles
                angle_hipshoulderelbow_left = self.calculate_angle(
                    hip_left, shoulder_left, elbow_left)
                angle_hipshoulderelbow_right = self.calculate_angle(
                    hip_right, shoulder_right, elbow_right)

                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                if angle_hipshoulderelbow_left > 48:
                    if angle_hipshoulderelbow_right < 45:
                        stance = "Right Hit"
                    else:
                        stance = "Stance OK"
                if angle_hipshoulderelbow_right > 48:
                    if angle_hipshoulderelbow_left < 45:
                        stance = "Left Hit"
                    else:
                        stance = "Stance OK"

            except Exception as e:
                print("Exception: ", e)
                pass

        return stance

    @staticmethod
    def check_for_invalid_pose(previous_states: list) -> bool:
        """
        Check if previous states were valid
        """
        for last_state in previous_states:
            if "None" in last_state["stance"]:
                return False
        return True

    @staticmethod
    def was_green_light_on(frame_number: int, paddle_results: dict) -> bool:
        """
        Checks if the green light was on in the last X frames
        """
        if frame_number < 30:
            return False
        for i in range(3):
            if paddle_results[frame_number - (i * 5)]:
                return True
        return False

    def evaluate_results(self, paddle_results: dict):
        """
        Iterate over the model results and filter out hit duplicates
        """
        results = []
        last_result = {"frame": -999, "stance": "None"}
        for i, state in enumerate(self.pose_states):
            if i < 5:
                continue
            if "Hit" not in state["stance"]:
                continue

            diff = state["frame"] - last_result["frame"]
            if diff < self.fps:
                continue

            if not self.check_for_invalid_pose(self.pose_states[i - 3:i]):
                continue

            if i < len(self.pose_states) - 3:
                if not self.check_for_invalid_pose(self.pose_states[i + 1:i + 2]):
                    continue

            result = state["stance"]
            if not self.was_green_light_on(state["frame"], paddle_results):
                result = "Invalid " + result
            else:
                result = "Valid " + result

            results.append({"time_code": self.get_timecode(
                state["frame"]), "result": result})
            last_result = state

        return results
