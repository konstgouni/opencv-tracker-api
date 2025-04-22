from flask import Flask, request, jsonify
import cv2
import uuid
import os
import requests

app = Flask(__name__)

@app.route("/track", methods=["POST"])
def track():
    print("✅ /track endpoint hit")
    data = request.json
    video_url = data.get("video_path")
    tracker_type = data.get("tracker", "MOSSE")
    bbox = data.get("bbox")

    if not (video_url and bbox):
        print("❌ Missing parameters")
        return jsonify({"error": "Missing video_path or bbox"}), 400

    try:
        print("📥 Downloading video...")
        local_video = f"input_{uuid.uuid4()}.mp4"
        r = requests.get(video_url, stream=True)
        with open(local_video, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        print("✅ Video downloaded:", local_video)
    except Exception as e:
        print("❌ Video download failed:", e)
        return jsonify({"error": f"Video download failed: {str(e)}"}), 500

    try:
        print("🎞️ Opening video file...")
        cap = cv2.VideoCapture(local_video)
        ok, frame = cap.read()
        if not ok:
            print("❌ Could not read first frame.")
            return jsonify({"error": "Failed to read video"}), 500
        print("✅ First frame read.")
    except Exception as e:
        print("❌ Video read error:", e)
        return jsonify({"error": f"Video read failed: {str(e)}"}), 500

    try:
        print(f"🧠 Creating tracker: {tracker_type}")
        if tracker_type.upper() == 'MOSSE':
            tracker = cv2.legacy.TrackerMOSSE_create()
        else:
            tracker = cv2.legacy.TrackerCSRT_create()

        tracker.init(frame, tuple(bbox))
        print("✅ Tracker initialized")
    except Exception as e:
        print("❌ Tracker init failed:", e)
        return jsonify({"error": f"Tracker failed: {str(e)}"}), 500

    cap.release()
    os.remove(local_video)
    print("🎉 All OK — returning dummy response")
    return jsonify({"status": "test success", "tracker": tracker_type})

# Local dev use only
if __name__ == "__main__":
    app.run(debug=True)


