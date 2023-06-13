from ultralytics import YOLO
from src.analyzer import Analyzer


class PaddleAnalysis(Analyzer):

    def __init__(self, video_path: str) -> None:

        super().__init__(video_path)
        self.paddle_states = []
    
        self.model = YOLO('models/paddletracker_v2.2.pt')
        self.threshold = 0.49

    def analyse(self) -> dict:
        print("Analysing paddle...")
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret or frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                break

            # Process every 5 frames
            if self.current_frame_number % 5 == 0:
                paddle_result = self.paddle_analysis(frame)
                if paddle_result is not None:
                    self.paddle_states.append(paddle_result)

            self.current_frame_number += 1
        return dict(self.paddle_states)
    
    def paddle_analysis(self, frame):
        result = self.model.predict(frame, self.threshold)
        result = list(result)  # Convert to a list
        boxes = result[0].boxes.xyxy
        scores = result[0].boxes.conf
        class_ids = result[0].names

        detected = False


        for _, score, class_id in zip(boxes, scores, class_ids):
            if class_id == 1:
                if score > self.threshold:
                    detected = True
        
        return [self.current_frame_number, detected]
