import cv2
import time
import torch
import math
from ultralytics import YOLO
import pyttsx3
import threading

# ---------------- DEVICE ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---------------- MODEL ----------------
model = YOLO("yolov8n.pt")
model.to(device)

VIDEO_PATH = r"C:\Users\Aditya Kurdekar\TrafficDetection\traffic.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

# ---------------- VOICE ----------------
engine = pyttsx3.init()

zebra_alert_time = 0
ZEBRA_COOLDOWN = 3

# 🔥 ZEBRA LINE (ABOVE CENTER)
ZEBRA_Y = 150

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------------- TRACKING ----------------
tracks = {}
track_id = 0

SPEED_LIMIT = 120
speed_memory = {}

# ---------------- TRACK ID ----------------
def get_track_id(cx, cy):
    global track_id, tracks

    for tid, (px, py, _) in tracks.items():
        if math.hypot(cx - px, cy - py) < 90:
            return tid

    tid = track_id
    track_id += 1
    return tid

# ---------------- CLASS FIX (CAR vs BIKE) ----------------
def refine_class(name, conf, w, h):

    aspect = w / (h + 1e-6)
    area = w * h

    if name == "person":
        if conf < 0.60:
            return None
        return "person"

    if name in ["motorcycle", "bicycle"]:
        return "bike"

    if name == "car":
        if area < 6000:
            return "bike"
        if aspect < 1.0:
            return "bike"
        return "car"

    if name in ["bus", "truck"]:
        return name

    return None

# ---------------- MAIN LOOP ----------------
while True:

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (900, 500))
    results = model.predict(frame, conf=0.5, device=device, verbose=False)

    car = 0
    bike = 0
    person = 0

    new_tracks = {}
    curr_time = time.time()

    zebra_person_detected = False

    # ---------------- ZEBRA LINE (INVISIBLE LOGIC ONLY) ----------------
    # (No drawing line as you asked)

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            name = model.names[cls].lower()

            w = x2 - x1
            h = y2 - y1

            name = refine_class(name, conf, w, h)

            if name is None:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            tid = get_track_id(cx, cy)

            # ---------------- SPEED ----------------
            speed = 0

            if tid in tracks:
                px, py, last_time = tracks[tid]

                dt = curr_time - last_time
                if dt < 0.001:
                    dt = 0.001

                speed = math.hypot(cx - px, cy - py) / dt

            new_tracks[tid] = (cx, cy, curr_time)

            # ---------------- ZEBRA DETECTION ----------------
            if name == "person" and abs(cy - ZEBRA_Y) < 70:
                zebra_person_detected = True

            # ---------------- COUNT ----------------
            if name == "car":
                car += 1
                color = (0, 255, 0)

            elif name == "bike":
                bike += 1
                color = (255, 0, 0)

            elif name == "person":
                person += 1
                color = (0, 255, 255)

            else:
                color = (200, 200, 200)

            # ---------------- DRAW ----------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, f"{name.upper()} {int(speed)}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ---------------- OVERSPEED (TEXT ONLY) ----------------
            if speed > SPEED_LIMIT:
                cv2.putText(frame, "OVERSPEED!",
                            (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)

    tracks = new_tracks

    # ---------------- ZEBRA SOUND ONLY ----------------
    if zebra_person_detected and (time.time() - zebra_alert_time > ZEBRA_COOLDOWN):
        zebra_alert_time = time.time()

        threading.Thread(
            target=speak,
            args=("Attention. Zebra crossing ahead. Please slow down.",),
            daemon=True
        ).start()

    # ---------------- UI ----------------
    cv2.putText(frame, f"CARS: {car}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"BIKES: {bike}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(frame, f"PEOPLE: {person}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"SPEED LIMIT: {SPEED_LIMIT}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("AI TRAFFIC SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()