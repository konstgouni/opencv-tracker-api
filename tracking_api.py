from flask import Flask, request, jsonify
import cv2
import os
import uuid
import urllib.request
import dropbox

DROPBOX_TOKEN = os.environ.get("DROPBOX_TOKEN")

app = Flask(__name__)

def upload_to_dropbox(local_path, dropbox_dest_path):
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)
    with open(local_path, 'rb') as f:
        dbx.files_upload(f.read(), dropbox_dest_path, mode=dropbox.files.WriteMode.overwrite)
    shared_link_metadata = dbx.sharing_create_shared_link_with_settings(dropbox_dest_path)
    return shared_link_metadata.url.replace('?dl=0', '?raw=1')

@app.route('/track', methods=['POST'])
def track():
    data = request.json
    video_path = data.get("video_path")
    tracker_type = data.get("tracker", "CSRT")
    bbox = data.get("bbox")

    if not (video_path and bbox):
        return jsonify({"error": "Missing video_path or bbox"}), 400

    local_video = f"input_{uuid.uuid4()}.mp4"
    try:
        urllib.request.urlretrieve(video_path, local_video)
    except:
        return jsonify({"error": "Failed to download video"}), 400

    # Create tracker
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy_TrackerBoosting.create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'CSRT':
        tracker = cv2.legacy_TrackerCSRT.create()
    elif tracker_type == 'TLD':
        tracker = cv2.legacy_TrackerTLD.create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy_TrackerMedianFlow.create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    else:
        tracker = cv2.legacy_TrackerMOSSE.create()

    cap = cv2.VideoCapture(local_video)
    ok, frame = cap.read()
    if not ok:
        return jsonify({"error": "Failed to read video"}), 500

    ok = tracker.init(frame, tuple(bbox))

    output_filename = f"tracked_{uuid.uuid4()}.mp4"
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame.shape[1], frame.shape[0]))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ok, bbox = tracker.update(frame)
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        out.write(frame)

    cap.release()
    out.release()

    dropbox_path = f"/{output_filename}"
    try:
        shared_url = upload_to_dropbox(output_filename, dropbox_path)
        os.remove(output_filename)
        os.remove(local_video)
        return jsonify({"status": "success", "video_url": shared_url})
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

