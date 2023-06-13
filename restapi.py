import argparse
from datetime import datetime

from flask import Flask, request, send_from_directory

from werkzeug.datastructures import FileStorage

from pathlib import Path

from src.pose_analysis import PoseAnalysis
from src.paddle_analysis import PaddleAnalysis

import shortuuid
app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
ENDPOINT_URL = "/api/analyse"


@app.route('/', methods=['GET'])
def index():
    """Serve the index HTML"""
    return send_from_directory('static', 'index.html')


@app.route(ENDPOINT_URL, methods=['POST'])
def analyse():
    """Analyse the video and return the results.

    """
    if not request.method == "POST":
        return "Method not allowed", 405

    files = request.files
    if len(files) != 2:
        return "Two video sources required.", 400

    start_time = datetime.now()
    file_path = Path(UPLOAD_FOLDER, shortuuid.uuid() +
                     files['poseVideo'].filename)
    files['poseVideo'].save(file_path)

    paddle_file_path = Path(
        UPLOAD_FOLDER, shortuuid.uuid() + files['paddleVideo'].filename)
    files['paddleVideo'].save(paddle_file_path)

    paddle_analysis = PaddleAnalysis(str(paddle_file_path))
    paddle_result = paddle_analysis.analyse()

    pose_analysis = PoseAnalysis(str(file_path))
    results = pose_analysis.analyse(paddle_result)

    print("Analysis took: {}".format(datetime.now() - start_time))
    return results, 200


def is_valid_file(file: FileStorage) -> bool:
    if file.extension not in ["mp4", "avi", "mov"]:
        raise ValueError("File extension not supported")
    if len(file) > 100000000:
        raise ValueError("File too large")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LUDUS API")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port)
