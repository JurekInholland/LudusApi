import cv2
import mediapipe as mp
import numpy as np


class PoseAnalysis:

    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.mp_pose = mp.solutions.pose
        self.current_frame_number = 0
        self.pose_states = []

    def analyse(self) -> list:
        print("Analysing pose...")
        # with self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
        #                        min_tracking_confidence=0.5) as pose:
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret or frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                break

            # Process every 5 frames
            if self.current_frame_number % 5 == 0:
                pose_result = self.pose_analysis(frame)
                self.pose_states.append({"frame": self.current_frame_number, "stance": pose_result})

            self.current_frame_number += 1

        results = self.evaluate_results()
        return results

    @staticmethod
    def calculate_angle(a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def pose_analysis(self, frame):
        with self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as pose:
            stance = "None"

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = pose.process(frame)

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
                angle_hipshoulderelbow_left = self.calculate_angle(hip_left, shoulder_left, elbow_left)
                angle_hipshoulderelbow_right = self.calculate_angle(hip_right, shoulder_right, elbow_right)

                if angle_hipshoulderelbow_left > 65:
                    if angle_hipshoulderelbow_right < 65:
                        print("Right hit")
                        stance = "Right hit"
                    else:
                        stance = "Stance OK"
                if angle_hipshoulderelbow_right > 65:
                    if angle_hipshoulderelbow_left < 65:
                        print("Left Hit")
                        stance = "Left hit"
                    else:
                        stance = "Stance OK"

            except Exception as e:
                print("Exception: ", e)
                pass

        return stance

    @staticmethod
    def previous_frames_ok(previous_states: list) -> bool:
        for last_state in previous_states:
            if "None" in last_state["stance"]:
                return False
        return True

    def get_timecode(self, current_frame):

        if self.fps == 0:
            return "00:00:00"

        # Calculate the current time in seconds
        current_time_in_seconds = current_frame / self.fps

        # Calculate hours, minutes, seconds and frames
        minutes, seconds = divmod(current_time_in_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        frames = current_frame - int(current_time_in_seconds) * self.fps

        # Format timecode as HH:MM:SS:FF
        timecode = "%02d:%02d:%02d" % (minutes, seconds, frames)

        return timecode

    def evaluate_results(self):
        results = []
        last_result = {"frame": -999, "stance": "None"}
        for i, state in enumerate(self.pose_states):
            if i < 5:
                continue
            if "hit" not in state["stance"]:
                continue

            diff = state["frame"] - last_result["frame"]
            if diff < self.fps:
                print("difference too low: ", diff)
                continue

            if not self.previous_frames_ok(self.pose_states[i - 5:i]):
                print("None in last 3: ", self.get_timecode(state["frame"]))
                continue

            results.append({"time_code": self.get_timecode(state["frame"]), "result": state["stance"]})
            last_result = state

        return results
