import cv2
import time
import torch
import math
import tkinter as tk
from ultralytics import YOLO
from PIL import Image, ImageTk
import pyttsx3
import threading


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


model = YOLO("yolov8n.pt")
model.to(device)

VIDEO_PATH = r"C:\Users\Aditya Kurdekar\TrafficDetection\traffic4.mp4"

cap = None
running = False 
tracks = {}
track_id = 0

SPEED_LIMIT = 120
speed_memory = {}
 

engine = pyttsx3.init()

zebra_alert_time = 0
ZEBRA_COOLDOWN = 3

ZEBRA_Y = 170  # slightly above center (fixed)

def speak(text):
    engine.say(text)
    engine.runAndWait()


def get_track_id(cx, cy):
    global tracks, track_id

    for tid, (px, py, _) in tracks.items():
        if math.hypot(cx - px, cy - py) < 90:
            return tid

    tid = track_id
    track_id += 1
    return tid


def refine_class(name, conf, w, h):

    aspect = w / (h + 1e-6)
    area = w * h

    if name == "person":
        if conf < 0.6:
            return None
        return "person"
    if name in ["motorcycle", "bicycle"]:
        return "bike"
    if name == "car":
        if area < 6000:
            return "bike"
        if aspect < 0.9:
            return "bike"
        return "car"

    if name in ["bus", "truck"]:
        return name

    return None
def start_video():
    global cap, running, tracks, track_id

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Video not found")
        return

    running = True
    tracks = {}
    track_id = 0

    process()
def stop_video():
    global running, cap
    running = False
    if cap:
        cap.release()
def process():
    global zebra_alert_time, tracks

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        root.after(10, process)
        return

    frame = cv2.resize(frame, (900, 500))

    results = model.predict(frame, conf=0.55, device=device, verbose=False)

    car = 0
    bike = 0
    person = 0

    new_tracks = {}
    curr_time = time.time()

    zebra_person_detected = False

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
            speed = 0

            if tid in tracks:
                px, py, last_time = tracks[tid]

                dt = curr_time - last_time
                if dt < 0.001:
                    dt = 0.001

                raw_speed = math.hypot(cx - px, cy - py) / dt

                speed_memory[tid] = 0.7 * speed_memory.get(tid, raw_speed) + 0.3 * raw_speed
                speed = speed_memory[tid]

            new_tracks[tid] = (cx, cy, curr_time)
            if name == "person" and abs(cy - ZEBRA_Y) < 70:
                zebra_person_detected = True
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
            if speed > SPEED_LIMIT:
                color = (0, 0, 255)  # RED BOX for overspeed

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, f"{name.upper()} {int(speed)}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if speed > SPEED_LIMIT:
                cv2.putText(frame, "OVERSPEED!",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

    tracks = new_tracks
    if zebra_person_detected and (time.time() - zebra_alert_time > ZEBRA_COOLDOWN):
        zebra_alert_time = time.time()

        threading.Thread(
            target=speak,
            args=("Attention. Zebra crossing ahead. Please slow down.",),
            daemon=True
        ).start()
    cv2.putText(frame, f"CARS: {car}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"BIKES: {bike}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(frame, f"PEOPLE: {person}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"SPEED LIMIT: {SPEED_LIMIT}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)

    label.imgtk = img
    label.configure(image=img)

    root.after(10, process)
root = tk.Tk()
root.title("🚦 AI Traffic Detection System")
root.geometry("950x780")
root.configure(bg="black")

label = tk.Label(root, bg="black")
label.pack()

tk.Button(root, text="START", command=start_video,
          bg="green", fg="white", width=15).pack(pady=5)

tk.Button(root, text="STOP", command=stop_video,
          bg="red", fg="white", width=15).pack(pady=5)
tk.Label(root,
         text="Under the Guidance of Sushma B R (CDS Department)",
         fg="cyan",
         bg="black",
         font=("Arial", 12, "bold")).pack(pady=10)
team = [
    "Team Member 1",
    "Team Member 2",
    "Team Member 3",
    "Team Member 4",
    "Team Member 5",
    "Team Member 6"
]

frame_team = tk.Frame(root, bg="black")
frame_team.pack()

tk.Label(frame_team, text="PROJECT TEAM",
         fg="cyan", bg="black",
         font=("Arial", 14, "bold")).pack()

for m in team:
    tk.Label(frame_team, text=m,
             fg="white", bg="black").pack()

root.mainloop()
