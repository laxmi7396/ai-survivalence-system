#code for AI SURVIVALENCE SYSTEM TO OBSTRUCT THE THINGS
from flask import Flask, render_template, Response, request, jsonify
import cv2
import imutils
import numpy as np
import os
import time
import threading
import winsound

app = Flask(_name_)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize tracker
TrDict = {
    'csrt': cv2.legacy.TrackerCSRT_create,
    'kcf': cv2.legacy.TrackerKCF_create,
    'boosting': cv2.legacy.TrackerBoosting_create,
    'mil': cv2.legacy.TrackerMIL_create,
    'tld': cv2.legacy.TrackerTLD_create,
    'medianflow': cv2.legacy.TrackerMedianFlow_create,
    'mosse': cv2.legacy.TrackerMOSSE_create
}
tracker = None
video_path = None
last_seen = time.time()
missing_threshold = 2  # Seconds before missing alert
object_lost = False
obstruction_count = 0  # Count obstruction occurrences
obstruction_start_time = 0  # Time when obstruction started

# Store tracking data
tracking_data = {
    "object_count": 0,
    "obstruction_time": None,
    "current_time": None,
    "status": "Idle"
}

# Function to play beep sound in parallel
def play_beep(frequency, duration, count):
    def sound_thread():
        for _ in range(count):
            winsound.Beep(frequency, duration)
            time.sleep(0.1)
    threading.Thread(target=sound_thread, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_path
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    video_path = file_path
    tracking_data["status"] = "File Tracking"
    return jsonify({'success': True})

@app.route('/start_file_tracking')
def start_file_tracking():
    return Response(track_objects(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_live_tracking')
def start_live_tracking():
    tracking_data["status"] = "Live Tracking"
    return Response(track_objects(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/tracking_data', methods=['GET'])
def get_tracking_data():
    tracking_data["current_time"] = time.strftime("%I:%M:%S %p")
    return jsonify(tracking_data)

@app.route('/api/update_obstruction', methods=['POST'])
def update_obstruction():
    global obstruction_count, obstruction_start_time
    data = request.get_json()
    obstruction_count = data.get('obstruction_count', obstruction_count)
    if obstruction_count > 0 and obstruction_start_time == 0:
        obstruction_start_time = time.time()
    elif obstruction_count == 0:
        obstruction_start_time = 0
    tracking_data.update({
        "object_count": obstruction_count,
        "obstruction_time": time.strftime("%I:%M:%S %p") if obstruction_start_time else None,
        "status": data.get('status', tracking_data["status"])
    })
    return jsonify({'success': True})

def track_objects(source):
    global tracker, last_seen, object_lost, obstruction_count, obstruction_start_time
    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    if not ret:
        return
    frame = imutils.resize(frame, width=900)
    tracker = TrDict['mosse']()
    bb = cv2.selectROI('Frame', frame, False)
    tracker.init(frame, bb)
    prev_x, prev_y = bb[0], bb[1]
    ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=900)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        success, box = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(a) for a in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 0), 2)
            movement_x = abs(x - prev_x) / w
            movement_y = abs(y - prev_y) / h

            if movement_x > 0.3 or movement_y > 0.3:
                play_beep(2500, 500, 3)

            diff = cv2.absdiff(ref_gray[y:y+h, x:x+w], gray[y:y+h, x:x+w])
            obstruction_level = np.mean(diff)

            if obstruction_level > 10:
                if obstruction_start_time == 0:
                    obstruction_start_time = time.time()
                obstruction_count += 1
                obstruction_duration = time.time() - obstruction_start_time

                if obstruction_duration > 10:
                    play_beep(2000, 500, 1)
                    obstruction_start_time = 0
                    obstruction_count = 0
            else:
                obstruction_start_time = 0

            last_seen = time.time()
            object_lost = False
            prev_x, prev_y = x, y
        else:
            missing_time = time.time() - last_seen
            if missing_time >= missing_threshold:
                object_lost = True
                play_beep(1500, 300, 20)

        if obstruction_start_time != 0:
            obstruction_count+=1
            cv2.putText(frame, f"Obstruction : {int(time.time() - obstruction_start_time)}s", (50, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

if _name_ == '_main_':
    app.run(debug=True, host='0.0.0.0')
  # front end part
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Object Tracking Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            min-height: 100vh;
            background: url('https://i.postimg.cc/1X4D8W4N/pexels-photo-430208.jpg') no-repeat center/cover;
            position: relative;
            overflow-x: hidden;
        }

        .background-shapes {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            opacity: 0.2;
        }

        .shape {
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.1);
            animation: float 15s infinite ease-in-out;
        }

        .shape:nth-child(1) { width: 100px; height: 100px; top: 20%; left: 10%; animation-delay: 0s; }
        .shape:nth-child(2) { width: 150px; height: 150px; top: 60%; right: 15%; animation-delay: 2s; }
        .shape:nth-child(3) { width: 80px; height: 80px; bottom: 10%; left: 30%; animation-delay: 4s; }

        header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(0px);
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        h2 {
            color: rgb(141, 74, 74);
            font-size: 2em;
            text-transform: uppercase;
            letter-spacing: 2px;
            text-shadow: 2px 2px 4px rgba(87, 163, 167, 0.2);
            animation: fadeIn 1s ease-in;
        }

        .container {
            padding: 40px;
            display: flex;
            justify-content: space-between;
            gap: 20px;
        }

        .left-panel {
            width: 20%;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .video-container {
            width: 75%;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease;
        }

        .video-container:hover {
            transform: scale(1.02);
        }

        .video-canvas {
            width: 100%;
            height: auto;
            max-height: 60vh;
            display: block;
        }

        .info-container {
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 10px;
            width: 90%;
            margin-top: 20px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }

        .info-item {
            margin: 10px 0;
            font-size: 0.9em;
        }

        .btn {
            padding: 12px 25px;
            font-size: 1.1em;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            background: #00b4d8;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(0, 180, 216, 0.4);
        }

        .btn:hover {
            background: #0077b6;
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0, 119, 182, 0.6);
        }

        input[type="file"] {
            display: none;
        }

        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-30px); }
            100% { transform: translateY(0px); }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 768px) {
            .container {
                flex-direction: column;
            }
            .left-panel, .video-container, .info-container {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="background-shapes">
        <div class="shape"></div>
        <div class="shape"></div>
        <div class="shape"></div>
    </div>

    <header>
        <h2>Real-Time Object Tracking</h2>
    </header>

    <div class="container">
        <div class="left-panel">
            <label class="btn" for="uploadInput">Upload Video</label>
            <input type="file" id="uploadInput" accept="video/*" onchange="uploadVideo()">
            <button class="btn" onclick="startLiveTracking()">Live Camera</button>
            <div class="info-container">
                <div class="info-item">Obstruction Time: <span id="obstructionTime">--</span></div>
                <div class="info-item">Current Time: <span id="currentTime">--</span></div>
                <div class="info-item">Object Count: <span id="objectCount">0</span></div>
                <div class="info-item">Status: <span id="status">Idle</span></div>
            </div>
        </div>
        
        <div class="video-container">
            <img id="videoFeed" src="" alt="Video Feed" style="display: none;">
        </div>
    </div>

    <script>
        function startLiveTracking() {
            const video = document.getElementById("videoFeed");
            video.src = "/start_live_tracking";
            video.style.display = "block";
            updateStatus("Live Tracking");
            updateObstructionData(0, "Live Tracking");
        }
        
        function uploadVideo() {
            const video = document.getElementById("videoFeed");
            let fileInput = document.getElementById("uploadInput");
            let formData = new FormData();
            formData.append("file", fileInput.files[0]);
            
            fetch("/upload_video", {
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    video.src = "/start_file_tracking";
                    video.style.display = "block";
                    updateStatus("File Tracking");
                    updateObstructionData(0, "File Tracking");
                } else {
                    alert("Error uploading file");
                }
            })
            .catch(error => {
                console.error("Upload failed:", error);
                alert("Upload failed!");
            });
        }

        function updateStatus(status) {
            document.getElementById("status").textContent = status;
        }

        function updateObstructionData(obstructionCount, status) {
            fetch("/api/update_obstruction", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    obstruction_count: obstructionCount,
                    status: status
                })
            })
            .catch(error => console.error("Error updating obstruction:", error));
        }

        function fetchTrackingData() {
            fetch("/api/tracking_data")
                .then(response => response.json())
                .then(data => {
                    document.getElementById("obstructionTime").textContent = data.obstruction_time || "--";
                    document.getElementById("currentTime").textContent = data.current_time;
                    document.getElementById("objectCount").textContent = data.object_count;
                    document.getElementById("status").textContent = data.status;
                })
                .catch(error => console.error("Error fetching tracking data:", error));
        }

        // Update tracking data every second
        setInterval(fetchTrackingData, 1000);

        // Initial fetch
        fetchTrackingData();
    </script>
</body>
</html> 
