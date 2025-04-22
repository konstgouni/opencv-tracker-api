from flask import Flask, request, jsonify
import cv2
import uuid
import os
import requests
import dropbox

app = Flask(__name__)

DROPBOX_TOKEN = os.environ.get("DROPBOX_TOKEN")

def upload_to_dropbox(file_path, dropbox_path):
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)
    with open(file_path, "rb") as f:
        dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)
    link = dbx.sharing_create_shared_link_with_settings(dropbox_path)
    return link.url.replace("?dl=0", "?raw=1")

def draw_rectangle(frame, bbox):
    p1 = (round(bbox[0]), round(bbox[1]))
    p2 = (round(bbox[0] + bbox[2]), round(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)

@app.route("/track", methods=["POST"])
def track():
    print("✅ /track called")
    data = request.json
    video_url = data.get("video_path")
    tracker_type = data.get("tracker", "MOSSE")
    bbox = data.get("bbox")

    if not (video_url and bbox):
        return jsonify({"error": "Missing video_path or bbox"}), 400

    try:
        print("📥 Downloading video...")
        local_video = f"input_{uuid.uuid4()}.mp4"
        r = requests.get(video_url, stream=True)
        with open(local_video, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        print("✅ Download complete:", local_video)
    except Exception as e:
        return jsonify({"error": f"Video download failed: {str(e)}"}), 500

    try:
        cap = cv2.VideoCapture(local_video)
        ok, frame = cap.read()
        if not ok:
            return jsonify({"error": "Could not read first frame"}), 500

        tracker = cv2.legacy.TrackerMOSSE_create()
        tracker.init(frame, tuple(bbox))

        output_video = f"tracked_{uuid.uuid4()}.mp4"
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame.shape[1], frame.shape[0]))

        max_frames = 30
        frame_count = 0

        while frame_count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            ok, bbox = tracker.update(frame)
            if ok:
                draw_rectangle(frame, bbox)
            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        os.remove(local_video)

        print("📤 Uploading result to Dropbox...")
        dropbox_path = f"/tracked/{output_video}"
        public_link = upload_to_dropbox(output_video, dropbox_path)
        os.remove(output_video)

        return jsonify({
            "status": "done",
            "frames_processed": frame_count,
            "video_url": public_link
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
