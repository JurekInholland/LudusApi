import cv2
import time
from ultralytics import YOLO
import mediapipe as mp
import numpy as np


async def analyse_videos(pose_video_path: str, paddle_video_path: str = ""):
    print("Analysing video: ", pose_video_path)
    model = YOLO('models/paddletracker v1.7.pt')
    pose_video = cv2.VideoCapture(pose_video_path)
    paddle_video = cv2.VideoCapture(paddle_video_path)
    breakpoint()
    paddle = video_analysis(paddle_video, model, "paddle")
    breakpoint()
    return video_analysis(pose_video, model, "pose")


def resize_video(input_path, output_path, width, height):
    video = cv2.VideoCapture(input_path)
    success, frame = video.read()
    if not success:
        raise ValueError("Kan de video niet lezen")

    # Krijg de oorspronkelijke breedte en hoogte van de video
    original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Bereken de schaalverhouding
    scale_ratio = min(width / original_width, height / original_height)

    # Bereken het nieuwe formaat
    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)

    # Maak een VideoWriter-object om het uitvoerbestand te maken
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (new_width, new_height))

    while success:
        # Verklein het frame naar het nieuwe formaat
        resized_frame = cv2.resize(frame, (new_width, new_height))
        # Schrijf het verkleinde frame naar het uitvoerbestand
        output_video.write(resized_frame)

        # Lees het volgende frame
        success, frame = video.read()

    # Sluit de video-objecten
    video.release()
    output_video.release()


def paddle_analysis(frame, model):
    threshold = 0.5
    class_names = ["Paddle", "Light_Green",
                   "Other class"]  # Voeg hier de namen van de klassen toe in de juiste volgorde

    result = model.predict(frame, threshold)
    result = list(result)  # Convert to a list
    boxes = result[0].boxes.xyxy.cuda()
    scores = result[0].boxes.conf.cuda()
    class_ids = result[0].names

    current_timestamp = time.time()
    detected = False

    # image = frame.copy()  # Create a copy of the frame

    for box, score, class_id in zip(boxes, scores, class_ids):
        if score >= threshold:
            box = [int(i) for i in box]

            if class_id < 2:
                # Draw bounding box on the image
                # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                # Add class label to the bounding box
                class_name = class_names[class_id]  # Get the class name from the list
                # label = f"{class_name}: {score:.2f}"
                # cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if class_id == 1:
                detected = True

    # Show the image with bounding boxes
    # cv2.imshow("Frame", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if detected:
        return [current_timestamp, 'legal_hit']
    else:
        return [current_timestamp, 'illegal_hit']


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def pose_analysis(frame, mp_pose):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Make detection

        stance = "Stance not OK"

        # Recolor back to BGR
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = pose.process(frame)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            # Calculate angles
            angle_hipshoulderelbow_left = calculate_angle(hip_left, shoulder_left, elbow_left)
            angle_hipshoulderelbow_right = calculate_angle(hip_right, shoulder_right, elbow_right)

            # Visualize angles
            # cv2.putText(frame, str(angle_hipshoulderelbow_left),
            #             tuple(np.multiply(shoulder_left, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            #
            # cv2.putText(frame, str(angle_hipshoulderelbow_right),
            #             tuple(np.multiply(shoulder_right, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Curl counter logic

            # print("LEFT", angle_hipshoulderelbow_left, " RIGHT", angle_hipshoulderelbow_right)
            stance = "None"
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

    return [time.time(), stance]


def get_timecode(cap, current_frame):
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        return "00:00:00"

    # Calculate the current time in seconds
    current_time_in_seconds = current_frame / fps

    # Calculate hours, minutes, seconds and frames
    minutes, seconds = divmod(current_time_in_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    frames = current_frame - int(current_time_in_seconds) * fps

    # Format timecode as HH:MM:SS:FF
    timecode = "%02d:%02d:%02d" % (minutes, seconds, frames)

    return timecode


def video_analysis(cap, model, mode):
    paddleState = []
    poseState = []

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret or frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                break

            # Process every 5 frames
            if frame_count % 5 == 0:
                if mode == 'pose':
                    poseResult = pose_analysis(frame, mp_pose)
                    poseState.append({"frame": frame_count, "stance": poseResult[1]})
                elif mode == 'paddle':
                    paddleState.append(paddle_analysis(frame, model))
                else:
                    raise ValueError(f"Invalid mode: {mode}")

            frame_count += 1

    if mode == 'pose':
        results = evaluate_results(cap, poseState)
        return results
    cap.release()
    return paddleState


def previous_frames_ok(previous_states) -> bool:
    for last_state in previous_states:
        if "None" in last_state["stance"]:
            return False
    return True


def evaluate_results(cap, states: list):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    results = []
    last_result = {"frame": -999, "stance": "None"}
    for i, state in enumerate(states):
        if i < 5:
            continue
        if "hit" not in state["stance"]:
            continue

        diff = state["frame"] - last_result["frame"]
        if diff < fps:
            print("difference too low: ", diff)
            continue

        if not previous_frames_ok(states[i - 5:i]):
            print("None in last 3: ", get_timecode(cap, state["frame"]))
            continue

        results.append({"time_code": get_timecode(cap, state["frame"]), "result": state["stance"]})
        last_result = state

    return results
