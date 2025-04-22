from flask import Flask, request, jsonify
import cv2
import os
import uuid
import urllib.request

app = Flask(__name__)

# Download GOTURN model if needed
def download_goturn_model():
    model_url = 'https://www.dropbox.com/scl/fi/gokz9sv1bczhsok8cpsiv/goturn.caffemodel?rlkey=hhc0b1b4jnu3053kdy9nisdbw&dl=1'
    prototxt_url = 'https://www.dropbox.com/scl/fi/cl2urkldmg7rss15haf4u/goturn.prototxt?rlkey=vnuods7f7y59e7xxeljanrtn4&dl=1'

    if not os.path.isfile('goturn.caffemodel'):
        urllib.request.urlretrieve(model_url, 'goturn.caffemodel')
    if not os.path.isfile('goturn.prototxt'):
        urllib.request.urlretrieve(prototxt_url, 'goturn.prototxt')

def drawRectangle(frame, bbox):
    p1 = (round(bbox[0]), round(bbox[1]))
    p2 = (round(bbox[0] + bbox[2]), round(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

def drawText(frame, txt, location, color = (50, 170, 50)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

@app.route('/track', methods=['POST'])
def track():
    data = request.json
    video_path = data.get("video_path")
    tracker_type = data.get("tracker", "CSRT")
    bbox = data.get("bbox")

    if not (video_path and bbox):
        return jsonify({"error": "Missing video_path or bbox"}), 400

    tmp_filename = f"video_{uuid.uuid4()}.mp4"
    try:
        urllib.request.urlretrieve(video_path, tmp_filename)
    except Exception as e:
        return jsonify({"error": f"Download failed: {str(e)}"}), 500

    if tracker_type == 'GOTURN':
        download_goturn_model()

    # Tracker selector
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
            tracker = cv2.TrackerGOTURN_create()
        else:
            tracker = cv2.legacy.TrackerMOSSE_create()
    except Exception as e:
        return jsonify({"error": "Tracker error: " + str(e)}), 500

    cap = cv2.VideoCapture(tmp_filename)
    ok, frame = cap.read()
    if not ok:
        return jsonify({"error": "Could not read video"}), 500

    ok = tracker.init(frame, tuple(bbox))

    frame_count = 0
    while frame_count < 20:
        ok, frame = cap.read()
        if not ok:
            break
        ok, bbox = tracker.update(frame)
        if ok:
            drawRectangle(frame, bbox)
        else:
            drawText(frame, "Tracking failed", (80, 140), (0, 0, 255))
        frame_count += 1

    cap.release()
    os.remove(tmp_filename)
    return jsonify({"status": "Success — processed 20 frames"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
