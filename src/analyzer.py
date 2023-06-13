import cv2

class Analyzer:
    """
    Base class for all analyzers.
    """
    
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_number = 0


    def get_timecode(self, current_frame: int) -> str:
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