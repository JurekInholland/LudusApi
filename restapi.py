import argparse
from flask import Flask, request
from src.yolo_analysis import analyse_videos

from werkzeug.datastructures import FileStorage
# import mediapipe as mp

from pathlib import Path
import shortuuid

from src.pose_analysis import PoseAnalysis

app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
ENDPOINT_URL = "/api/analyse"


@app.route('/', methods=['GET'])
def index():
    return "LUDUS API is running.", 200


@app.route(ENDPOINT_URL, methods=['POST'])
def analyse():
    if not request.method == "POST":
        return "Method not allowed", 405

    files = request.files
    if len(files) != 2:
        return "Two video sources required.", 400

    # f_test = files[0]
    # f_test2 = files[1]
    # breakpoint()

    file_path = Path(UPLOAD_FOLDER, files['file1'].filename)
    files['file1'].save(file_path)

    paddle_file_path = Path(UPLOAD_FOLDER, files['file2'].filename)
    files['file2'].save(paddle_file_path)

    # result = await analyse_videos(str(file_path), str(paddle_file_path))

    pose_analysis = PoseAnalysis(str(file_path))
    pose_result = pose_analysis.analyse()

    return pose_result, 200


def is_valid_file(file: FileStorage) -> bool:
    if file.extension not in ["mp4", "avi", "mov"]:
        raise ValueError("File extension not supported")
        # return False
    if len(file) > 100000000:
        raise ValueError("File too large")
        # return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LUDUS API")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port)
