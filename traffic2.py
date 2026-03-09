import cv2
import numpy as np
import time
import os
import serial
from ultralytics import YOLO

# ---------------------------
# Master Configuration
# ---------------------------
VIDEO_SOURCES = [
    {"path": "demo.mp4", "roi": ((150, 100), (620, 700)), "label": "Road 1 (High)"}, # Adjusted ROI
    {"path": "demo2.mp4", "roi": ((100, 100), (580, 680)), "label": "Road 2 (V. High)"},
    {"path": "road_video3.mp4", "roi": ((50, 100), (300, 400)), "label": "Road 3 (Medium)"},
    {"path": "road_video4.mp4", "roi": ((50, 100), (300, 400)), "label": "Road 4 (Low)"}
]
ARDUINO_PORT = "COM6"
RUSH_HOUR_THRESHOLD = 30
NORMAL_THRESHOLD = 10
NIGHT_TIME_THRESHOLD = 1
MODEL_NAME = 'yolov8s.pt'
# --- ADJUSTED FOR STABILITY ---
NTH_FRAME = 30 # Process detection even less frequently
VEHICLE_CLASS_IDS = [2, 3, 5, 7] # car, motorcycle, bus, truck
DETECTION_CONFIDENCE = 0.4

# --- Optimization & Display Settings ---
YOLO_INPUT_SIZE = 640
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360
WINDOW_NAME = "YOLOv8 Vehicle Detection - Automated Command Center (720p Optimized)"

# --- Scenario Timings ---
SCENARIO_GREEN_TIMES_MS = {
    '1': [20000, 20000, 20000, 20000], # Normal
    '2': [40000, 10000, 40000, 10000], # Rush Hour
    '3': [10000, 10000, 10000, 10000], # Night
    '4': [0, 0, 0, 0]                  # All Red
}
DEFAULT_SCENARIO_COMMAND = '1'

# --- Stability Control ---
# --- INCREASED CONFIRMATION CYCLES ---
CONFIRMATION_CYCLES_NEEDED = 10 # Require 10 consistent readings
potential_next_command = None
confirmation_counter = 0

# ---------------------------
# Initialization (Unchanged)
# ---------------------------
try:
    arduino = serial.Serial(port=ARDUINO_PORT, baudrate=9600, timeout=0.1)
    print(f"Arduino connected on {ARDUINO_PORT}!")
    time.sleep(2)
except Exception as e:
    print(f"Error connecting to Arduino: {e}. Running in detection-only mode.")
    arduino = None

try:
    model = YOLO(MODEL_NAME)
    print(f"YOLOv8 model ({MODEL_NAME}) loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    exit()

print("Initializing video streams...")
for source in VIDEO_SOURCES:
    if not os.path.exists(source['path']):
        print(f"Warning: Video file not found: {source['path']}.")
        source['active'] = False; source['cap'] = None
    else:
        source['cap'] = cv2.VideoCapture(source['path'])
        if not source['cap'].isOpened():
            print(f"Warning: Could not open video: {source['path']}")
            source['active'] = False
        else:
            source['active'] = True
    source['last_count'] = 0
    source['last_detections'] = []
print("Initialization complete.")

# ---------------------------
# Core Functions (Unchanged - create_placeholder, detect_vehicles, send_command)
# ---------------------------
def create_placeholder_frame(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, text="No Signal"):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_x = (width - text_width) // 2
    text_y = (height + text_height) // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame

def detect_vehicles_in_roi(frame_original, roi_top_left, roi_bottom_right):
    orig_h, orig_w = frame_original.shape[:2]
    input_frame = cv2.resize(frame_original, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
    roi_count = 0
    detected_vehicles = []
    results = model.predict(input_frame, classes=VEHICLE_CLASS_IDS, conf=DETECTION_CONFIDENCE, verbose=False)
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy(); confs = result.boxes.conf.cpu().numpy(); class_ids = result.boxes.cls.cpu().numpy()
        scale_x = orig_w / YOLO_INPUT_SIZE; scale_y = orig_h / YOLO_INPUT_SIZE
        for i in range(len(boxes)):
            x1_pred, y1_pred, x2_pred, y2_pred = boxes[i]
            x1 = int(x1_pred * scale_x); y1 = int(y1_pred * scale_y); x2 = int(x2_pred * scale_x); y2 = int(y2_pred * scale_y)
            conf = confs[i]; cls_id = int(class_ids[i]); label = model.names[cls_id]
            w, h = x2 - x1, y2 - y1
            detected_vehicles.append(((x1, y1, w, h), label, conf))
            box_center_x = x1 + w / 2; box_center_y = y1 + h / 2
            if (roi_top_left[0] < box_center_x < roi_bottom_right[0] and roi_top_left[1] < box_center_y < roi_bottom_right[1]):
                roi_count += 1
    return roi_count, detected_vehicles

def send_command_to_arduino(counts, last_sent_command):
    global potential_next_command, confirmation_counter # Use global state variables

    # 1. Determine the scenario indicated by current counts
    if not counts: max_count = 0
    else: max_count = max(counts)

    indicated_command = None
    if max_count >= RUSH_HOUR_THRESHOLD: indicated_command = '2'
    elif max_count >= NORMAL_THRESHOLD: indicated_command = '1'
    else: indicated_command = '3'

    if indicated_command is None:
        indicated_command = last_sent_command if last_sent_command else DEFAULT_SCENARIO_COMMAND

    # 2. Implement Stability Logic
    if indicated_command == last_sent_command:
        confirmation_counter = 0
        potential_next_command = None
        return last_sent_command # No change needed
    else:
        if indicated_command == potential_next_command:
            confirmation_counter += 1
        else:
            potential_next_command = indicated_command
            confirmation_counter = 1

        print(f"    -> Potential new command: '{potential_next_command}', Seen: {confirmation_counter}/{CONFIRMATION_CYCLES_NEEDED} times.") # Debug print

        # 3. Check if confirmation threshold is met
        if confirmation_counter >= CONFIRMATION_CYCLES_NEEDED:
            if arduino:
                try:
                    arduino.write(bytes(potential_next_command, 'utf-8'))
                    print(f"[{time.strftime('%H:%M:%S')}] CONFIRMED New Scenario! Sending: '{potential_next_command}' (Max Count: {max_count})")
                except Exception as e: print(f"Serial Error on {ARDUINO_PORT}: {e}")
            else: print(f"[{time.strftime('%H:%M:%S')}] CONFIRMED New Scenario (No Arduino): '{potential_next_command}' (Max Count: {max_count})")

            confirmed_command = potential_next_command
            potential_next_command = None
            confirmation_counter = 0
            return confirmed_command
        else:
            return last_sent_command # Not confirmed yet

# ---------------------------
# Main Loop (Unchanged drawing logic)
# ---------------------------
print(f"Starting video processing system with YOLOv8 ({MODEL_NAME})...")
frame_count = 0
last_sent_command = None

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH * 2, DISPLAY_HEIGHT * 2)

while True:
    start_time_frame = time.time()
    active_streams = 0
    frames_original = []
    for source in VIDEO_SOURCES:
        if source.get('active', False):
            ret, frame = source['cap'].read()
            if ret: frames_original.append(frame); active_streams += 1
            else:
                source['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = source['cap'].read()
                if ret: frames_original.append(frame); active_streams += 1
                else: frames_original.append(None); print(f"Warning: Failed loop {source['path']}")
        else: frames_original.append(None)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

    frame_count += 1
    if frame_count % NTH_FRAME == 0:
        current_counts = []
        for i, source in enumerate(VIDEO_SOURCES):
            if source.get('active', False) and frames_original[i] is not None:
                count, detections = detect_vehicles_in_roi(frames_original[i], source['roi'][0], source['roi'][1])
                source['last_count'] = count
                source['last_detections'] = detections
                current_counts.append(count)
            else:
                current_counts.append(0)
                source['last_detections'] = []
        last_sent_command = send_command_to_arduino(current_counts, last_sent_command)

    display_frames = []
    active_scenario_command = last_sent_command if last_sent_command else DEFAULT_SCENARIO_COMMAND
    current_timings_ms = SCENARIO_GREEN_TIMES_MS.get(active_scenario_command, SCENARIO_GREEN_TIMES_MS[DEFAULT_SCENARIO_COMMAND])

    for i, source in enumerate(VIDEO_SOURCES):
        if source.get('active', False) and frames_original[i] is not None:
            draw_frame = frames_original[i].copy()
        else:
            draw_frame = create_placeholder_frame(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, text=f"Road {i+1}\nInactive")
            display_frames.append(draw_frame)
            continue

        if 'last_detections' in source:
             for box_info, label, conf in source['last_detections']:
                (x, y, w, h) = box_info
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(draw_frame.shape[1], x + w), min(draw_frame.shape[0], y + h)
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label} {conf:.2f}"
                    cv2.putText(draw_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if 'roi' in source:
            roi_tl, roi_br = source['roi']
            roi_x1, roi_y1 = max(0, roi_tl[0]), max(0, roi_tl[1])
            roi_x2, roi_y2 = min(draw_frame.shape[1], roi_br[0]), min(draw_frame.shape[0], roi_br[1])
            if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                 cv2.rectangle(draw_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
                 label_text = source.get('label', f'Road {i+1}')
                 count_text = f"{label_text}: {source.get('last_count', 0)}"
                 text_x, text_y = max(roi_x1, 0) + 5, max(roi_y1, 0) - 10
                 cv2.putText(draw_frame, count_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                 green_time_sec = current_timings_ms[i] // 1000
                 time_text = f"Green: {green_time_sec}s"
                 cv2.putText(draw_frame, time_text, (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        display_frame_resized = cv2.resize(draw_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        display_frames.append(display_frame_resized)

    while len(display_frames) < 4:
        display_frames.append(create_placeholder_frame(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT))
    top_row = np.hstack((display_frames[0], display_frames[1]))
    bottom_row = np.hstack((display_frames[2], display_frames[3]))
    combined_frame = np.vstack((top_row, bottom_row))

    end_time_frame = time.time()
    fps = 1 / (end_time_frame - start_time_frame) if (end_time_frame - start_time_frame) > 0 else 0
    cv2.putText(combined_frame, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    scenario_names = {'1': "Normal Flow", '2': "Rush Hour", '3': "Night Time", '4': "ALL RED"}
    active_scenario_name = scenario_names.get(active_scenario_command, "Unknown")
    cv2.putText(combined_frame, f"Scenario: {active_scenario_name}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, combined_frame)

# ---------------------------
# Cleanup
# ---------------------------
print("Cleaning up...")
if arduino:
    try: arduino.write(bytes('4', 'utf-8')) # Send 'All Red'
    except: pass
    arduino.close()
for source in VIDEO_SOURCES:
    if source.get('cap'): source['cap'].release()
cv2.destroyAllWindows()

