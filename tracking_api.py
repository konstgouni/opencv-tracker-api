from flask import Flask, request, jsonify
import cv2
import uuid
import os
import requests
import dropbox
import urllib.request

app = Flask(__name__)

# 🔐 From Render's Environment Variables
DROPBOX_TOKEN = os.environ.get("DROPBOX_TOKEN")

# ✅ GOTURN Model Downloader
def download_goturn_model():
    model_url = "https://www.dropbox.com/scl/fi/gokz9sv1bczhsok8cpsiv/goturn.caffemodel?rlkey=hhc0b1b4jnu3053kdy9nisdbw&dl=1"
    prototxt_url = "https://www.dropbox.com/scl/fi/cl2urkldmg7rss15haf4u/goturn.prototxt?rlkey=vnuods7f7y59e7xxeljanrtn4&dl=1"

    if not os.path.exists("goturn.caffemodel"):
        print("📥 Downloading goturn.caffemodel...")
        urllib.request.urlretrieve(model_url, "goturn.caffemodel")

    if not os.path.exists("goturn.prototxt"):
        print("📥 Downloading goturn.prototxt...")
        urllib.request.urlretrieve(prototxt_url, "goturn.prototxt")

# ✅ Upload tracked video to Dropbox
def upload_to_dropbox(file_path, dropbox_path):
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)
    with open(file_path, "rb") as f:
        dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)
    link = dbx.sharing_create_shared_link_with_settings(dropbox_path)
    return link.url.replace("?dl=0", "?raw=1")

# ✅ Drawing utilities
def draw_rectangle(frame, bbox):
    p1 = (round(bbox[0]), round(bbox[1]))
    p2 = (round(bbox[0] + bbox[2]), round(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)

def draw_text(frame, text, location, color=(50, 170, 50)):
    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# ✅ API Endpoint
@app.route("/track", methods=["POST"])
def track():
    data = request.json
    video_url = data.get("video_path")
    tracker_type = data.get("tracker", "CSRT")
    bbox = data.get("bbox")

    if not (video_url and bbox):
        return jsonify({"error": "Missing video_path or bbox"}), 400

    # 🔽 Download the video from URL
    try:
        local_video = f"input_{uuid.uuid4()}.mp4"
        r = requests.get(video_url, stream=True)
        with open(local_video, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
    except Exception as e:
        return jsonify({"error": f"Video download failed: {str(e)}"}), 500

    # ✅ Tracker Selection
    try:
        if tracker_type == 'BOOSTING':
            tracker = cv2.legacy.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == 'CSRT':
            tracker = cv2.legacy.TrackerCSRT_create()
        elif tracker_type == 'TLD':
            tracker = cv2.legacy.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv2.legacy.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
            download_goturn_model()
            tracker = cv2.TrackerGOTURN_create()
        else:
            tracker = cv2.legacy.TrackerMOSSE_create()
    except Exception as e:
        return jsonify({"error": f"Tracker init failed: {str(e)}"}), 500

    cap = cv2.VideoCapture(local_video)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        os.remove(local_video)
        return jsonify({"error": "Failed to read video"}), 500

    tracker.init(frame, tuple(bbox))

    output_video = f"tracked_{uuid.uuid4()}.mp4"
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame.shape[1], frame.shape[0]))

    frame_count = 0
    max_frames = 50  # Adjust if needed to avoid Render timeouts

    while frame_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        ok, bbox = tracker.update(frame)
        if ok:
            draw_rectangle(frame, bbox)
        else:
            draw_text(frame, "Tracking failed", (50, 80), (0, 0, 255))
        draw_text(frame, f"{tracker_type} Tracker", (50, 50))
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    os.remove(local_video)

    try:
        dropbox_url = upload_to_dropbox(output_video, f"/tracked/{output_video}")
        os.remove(output_video)
        return jsonify({
            "status": "done",
            "tracker": tracker_type,
            "frames_processed": frame_count,
            "video_url": dropbox_url
        })
    except Exception as e:
        return jsonify({"error": f"Dropbox upload failed: {str(e)}"}), 500

# ✅ Run local (for testing, won't run on Render)
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

