from cv2 import rectangle, VideoCapture, TrackerKCF_create, legacy_TrackerMedianFlow, VideoWriter, VideoWriter_fourcc, CAP_PROP_FPS
from dropbox import Dropbox
from dropbox.files import WriteMode
from flask import Flask, request, jsonify
from flask_cors import CORS
from uuid import uuid4
from os import remove, environ
from requests import get
# Global scope variables and objects are defined below:
DROPBOX_TOKEN, app = environ.get("DROPBOX_TOKEN"), Flask(__name__)
CORS(app)
# The method below aims at uploading the processed image / sequence video signal to Dropbox
def upload_to_dropbox(file_path, dropbox_path):
    dbx = Dropbox(DROPBOX_TOKEN)
    with open(file_path, "rb") as f:
        dbx.files_upload(f.read(), dropbox_path, mode = WriteMode.overwrite)
    link = dbx.sharing_create_shared_link_with_settings(dropbox_path)
    return link.url.replace("?dl=0", "?raw=1")
# The method below aims at drawing the bounding box on the successfully tracked object in the video signal
def draw_rectangle(frame, bbox):
    p1 = (round(bbox[0]), round(bbox[1]))
    p2 = (round(bbox[0] + bbox[2]), round(bbox[1] + bbox[3]))
    rectangle(frame, p1, p2, (0, 0, 255), 2)
# Main object tracking functionality of the hosted application is defined just below
@app.route("/track", methods = ["POST"])
def track():
    print("Object Tracker Successfully Called")
    data = request.json
    video_url, tracker_type, bbox = data.get("video_path"), data.get("tracker"), data.get("bbox")
    if not (video_url and bbox):
        return jsonify({"error": "Missing video_path or initial object bounding box prediction"}), 400
    try:
        print("Downloading video..")
        local_video = f"input_{uuid4()}.mp4"
        r = get(video_url, stream = True)
        with open(local_video, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        print("Video dwnload complete:", local_video)
    except Exception as e:
        return jsonify({"error": f"Video download failed: {str(e)}"}), 500
    try:
        cap = VideoCapture(local_video)
        ok, frame = cap.read()
        if not ok:
            return jsonify({"error": "Could not read the first frame"}), 500
        if tracker_type == "KCF":
            tracker = TrackerKCF_create()
            print('The Kernelized Correlation Filter (KCF) Object Tracker will be used!')
        else:
            print('A non-default tracker might have been set. Due to execution performace related constraints the MEDIANFLOW tracker will be used instead')
            tracker = legacy_TrackerMedianFlow.create()
        tracker.init(frame, tuple(bbox))
        output_video = f"tracked_{uuid4()}.mp4"
        targetFPS = cap.get(CAP_PROP_FPS) or 30.0
        out = VideoWriter(output_video, VideoWriter_fourcc(*'mp4v'), targetFPS, (frame.shape[1], frame.shape[0]))
        max_frames = 200
        frame_count = 0
        while frame_count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            trackerSuccess, bbox = tracker.update(frame)
            if trackerSuccess:
                draw_rectangle(frame, bbox)
            out.write(frame)
            frame_count += 1
        cap.release()
        out.release()
        remove(local_video)
        print("Uploading Object Tracking Results to Dropbox...")
        dropbox_path = f"/tracked/{output_video}"
        try:
            public_link = upload_to_dropbox(output_video, dropbox_path)
            remove(output_video)
        except Exception as e:
            print("Dropbox upload unfortunately failed:", str(e))
            return jsonify({"error": f"Dropbox upload failed: {str(e)}"}), 500
        # successfull program exit with video signal processing results lies on the line below:
        return jsonify({"status": "done", "frames_processed": frame_count, "video_url": public_link})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Main method below, for retaining good coding practices
if __name__ == "__main__":
    app.run(debug = True)
