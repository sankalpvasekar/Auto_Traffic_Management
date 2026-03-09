import cv2
import numpy as np
import time
import serial
from ultralytics import YOLO

# YOLO Configuration (migrated to YOLOv8)
MODEL_NAME = 'yolov8s.pt'  
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
DETECTION_CONFIDENCE = 0.35
YOLO_INPUT_SIZE = 768

# Counting thresholds (globals for easy tuning)
OVERLAP_FRAC_THRESH = 0.25   # at least 25% of box must lie inside ROI to count
MIN_BBOX_AREA_FRAC = 0.001   # ignore very small boxes (<0.1% of frame area)

try:
    model = YOLO(MODEL_NAME)
except Exception as e:
    print(f"Error loading YOLO model {MODEL_NAME}: {e}")
    raise

# Arduino Serial Configuration
ARDUINO_PORT = "COM6"
BAUD_RATE = 9600

arduino = None
try:
    arduino = serial.Serial(port=ARDUINO_PORT, baudrate=BAUD_RATE, timeout=0.1)
    time.sleep(2)
except Exception as e:
    print(f"Warning: Could not open Arduino on {ARDUINO_PORT}: {e}. Running without hardware.")
    arduino = None

# ---------------------------
# Video Capture Initialization
# ---------------------------
cap1 = cv2.VideoCapture(r"C:\Users\sanka\OneDrive\Desktop\Project\FrontVehcilesTraffic1080p.mp4")
cap2 = cv2.VideoCapture(r"C:\Users\sanka\OneDrive\Desktop\Project\sampleVideos\demo3.mp4")
cap3 = cv2.VideoCapture(r"C:\Users\sanka\OneDrive\Desktop\Project\sampleVideos\Video.mp4")
cap4 = cv2.VideoCapture(r"C:\Users\sanka\OneDrive\Desktop\Project\road_video4 - Trim.mp4")

roi_cam3 = ((50, 100), (300, 400))   # top-left, bottom-right for camera 3
roi_cam4 = ((50, 100), (300, 400)) 
roi_cam1 = ((200, 100), (500, 600))   # top-left, bottom-right for camera 1
roi_cam2 = ((100, 100), (580, 680))   # top-left, bottom-right for camera 2

# Process every nth frame (for efficiency)
nth_frame = 15
frame_count = 0

# Store last detected counts to prevent flickering (one per camera)
last_roi_counts = [0, 0, 0, 0]  # [cap1, cap2, cap3, cap4]

def _center_roi_with_bias(frame, wf=0.45, hf=0.55, y_bias_frac=0.12):
    """Compute a centered ROI with optional downward bias (towards the near lane)."""
    H, W = frame.shape[:2]
    w = max(1, int(W * wf))
    h = max(1, int(H * hf))
    cx = W // 2
    cy = H // 2 + int(H * y_bias_frac)
    x1 = max(0, cx - w // 2)
    y1 = max(0, cy - h // 2)
    x2 = min(W, x1 + w)
    y2 = min(H, y1 + h)
    # Clamp if bias pushed beyond bounds
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        x1, y1 = max(0, (W - w) // 2), max(0, (H - h) // 2)
        x2, y2 = min(W, x1 + w), min(H, y1 + h)
    return (x1, y1), (x2, y2)

roi1_inited = False
roi2_inited = False
roi3_inited = False
roi4_inited = False

def detect_vehicles_in_roi(frame, roi_top_left, roi_bottom_right):
    """
    Run YOLOv8 detection on the frame and count vehicles whose centers fall inside the ROI.
    Also draw boxes and labels for visualization.
    """
    orig_h, orig_w = frame.shape[:2]
    input_frame = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))

    roi_count = 0
    results = model.predict(input_frame, classes=VEHICLE_CLASS_IDS, conf=DETECTION_CONFIDENCE, verbose=False)
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        scale_x = orig_w / YOLO_INPUT_SIZE
        scale_y = orig_h / YOLO_INPUT_SIZE
        roi_x1, roi_y1 = roi_top_left
        roi_x2, roi_y2 = roi_bottom_right
        for i in range(len(boxes)):
            x1p, y1p, x2p, y2p = boxes[i]
            x1 = int(x1p * scale_x); y1 = int(y1p * scale_y)
            x2 = int(x2p * scale_x); y2 = int(y2p * scale_y)
            # clamp
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(orig_w, x2); y2 = min(orig_h, y2)
            w = max(1, x2 - x1); h = max(1, y2 - y1)
            box_area = w * h
            # min area filter (relative to frame)
            if box_area < MIN_BBOX_AREA_FRAC * (orig_w * orig_h):
                # draw excluded in gray
                cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
                continue

            # overlap with ROI (rectangle)
            ix1 = max(roi_x1, x1); iy1 = max(roi_y1, y1)
            ix2 = min(roi_x2, x2); iy2 = min(roi_y2, y2)
            inter_w = max(0, ix2 - ix1); inter_h = max(0, iy2 - iy1)
            inter_area = inter_w * inter_h
            overlap_frac = inter_area / float(box_area)

            # draw
            counted = overlap_frac >= OVERLAP_FRAC_THRESH
            color = (0, 255, 0) if counted else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            try:
                class_name = model.names[int(class_ids[i])]
            except Exception:
                class_name = str(int(class_ids[i]))
            cv2.putText(frame, f"{class_name} {confs[i]:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # Mark center if counted
            if counted:
                cx = x1 + w // 2; cy = y1 + h // 2
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            if counted:
                roi_count += 1

    # Draw ROI and count
    cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 255, 255), 2)
    cv2.putText(frame, f"ROI Count: {roi_count}", (roi_top_left[0], roi_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return roi_count

# ---------------------------
# Decision Logic and Serial Protocol
# ---------------------------
MIN_GREEN_MS = 5000
MAX_GREEN_MS = 45000
PER_VEHICLE_MS = 1000

waiting_for_cycle_end = False
active_road_index = None
cycle_deadline_monotonic = 0.0

def compute_green_time_ms(count):
    duration = MIN_GREEN_MS + int(count) * PER_VEHICLE_MS
    return max(MIN_GREEN_MS, min(MAX_GREEN_MS, duration))

def send_green_command(road_index, duration_ms):
    # Protocol: "G,<road 1-4>,<duration_ms>\n"
    payload = f"G,{road_index + 1},{int(duration_ms)}\n"
    if arduino:
        try:
            arduino.write(payload.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Serial write failed on {ARDUINO_PORT}: {e}")
            return False
    else:
        print(f"[SIM] Would send: {payload.strip()}")
        return True

def check_arduino_done():
    if not arduino:
        return False
    try:
        line = arduino.readline().decode('utf-8', errors='ignore').strip()
        if line:
            if line.upper().startswith("DONE"):
                return True
    except Exception:
        pass
    return False

while True:
    frame_count += 1

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

    # Break loop if any video stream ends
    if not ret1 or not ret2 or not ret3 or not ret4:
        break

    # Initialize centered-biased ROIs once per road using current frame sizes
    if ret1 and not roi1_inited:
        roi_cam1 = _center_roi_with_bias(frame1, wf=0.45, hf=0.55, y_bias_frac=0.12)
        roi1_inited = True
    if ret2 and not roi2_inited:
        roi_cam2 = _center_roi_with_bias(frame2, wf=0.45, hf=0.55, y_bias_frac=0.12)
        roi2_inited = True
    if ret3 and not roi3_inited:
        roi_cam3 = _center_roi_with_bias(frame3, wf=0.45, hf=0.55, y_bias_frac=0.12)
        roi3_inited = True
    if ret4 and not roi4_inited:
        roi_cam4 = _center_roi_with_bias(frame4, wf=0.45, hf=0.55, y_bias_frac=0.12)
        roi4_inited = True

    # Draw ROI boxes on all frames (to avoid flickering)
    cv2.rectangle(frame1, roi_cam1[0], roi_cam1[1], (0, 255, 255), 2)
    cv2.rectangle(frame2, roi_cam2[0], roi_cam2[1], (0, 255, 255), 2)
    cv2.rectangle(frame3, roi_cam3[0], roi_cam3[1], (0, 255, 255), 2)
    cv2.rectangle(frame4, roi_cam4[0], roi_cam4[1], (0, 255, 255), 2)

    # If we are not in an active green cycle, run detection and possibly start a new cycle
    if not waiting_for_cycle_end and frame_count % nth_frame == 0:
        # Camera 1
        last_roi_counts[0] = detect_vehicles_in_roi(frame1, roi_cam1[0], roi_cam1[1])
        # Camera 2
        last_roi_counts[1] = detect_vehicles_in_roi(frame2, roi_cam2[0], roi_cam2[1])
        # Camera 3
        last_roi_counts[2] = detect_vehicles_in_roi(frame3, roi_cam3[0], roi_cam3[1])
        # Camera 4
        last_roi_counts[3] = detect_vehicles_in_roi(frame4, roi_cam4[0], roi_cam4[1])

        # Persist counts to file
        with open("carsCount.txt", "w") as f:
            f.write(f"{last_roi_counts[0]} {last_roi_counts[1]} {last_roi_counts[2]} {last_roi_counts[3]}")

        # Decide which road gets green
        active_road_index = int(np.argmax(last_roi_counts))
        green_ms = compute_green_time_ms(last_roi_counts[active_road_index])

        ok = send_green_command(active_road_index, green_ms)
        waiting_for_cycle_end = ok
        cycle_deadline_monotonic = time.monotonic() + (green_ms / 1000.0) + 12.0  # +yellow+intergreen buffer

    # Always display the last detected ROI count (no flickering)
    cv2.putText(frame1, f"ROI Count: {last_roi_counts[0]}",
                (roi_cam1[0][0], roi_cam1[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame2, f"ROI Count: {last_roi_counts[1]}",
                (roi_cam2[0][0], roi_cam2[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame3, f"ROI Count: {last_roi_counts[2]}",
                (roi_cam3[0][0], roi_cam3[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame4, f"ROI Count: {last_roi_counts[3]}",
                (roi_cam4[0][0], roi_cam4[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Prepare frames list for selective overlays
    frames = [frame1, frame2, frame3, frame4]

    # Overlay active decision if in-cycle
    if waiting_for_cycle_end and active_road_index is not None:
        labels = ["Road 1", "Road 2", "Road 3", "Road 4"]
        status_text = f"ACTIVE GREEN: {labels[active_road_index]}"
        # Only show the green status on the selected road's frame
        sel_frame = frames[active_road_index]
        cv2.putText(sel_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        # Check for Arduino DONE or fallback timeout
        if check_arduino_done() or time.monotonic() >= cycle_deadline_monotonic:
            waiting_for_cycle_end = False
            active_road_index = None

    # Resize frames for a 2×2 grid display
    frame1_resized = cv2.resize(frame1, (640, 480))
    frame2_resized = cv2.resize(frame2, (640, 480))
    frame3_resized = cv2.resize(frame3, (640, 480))
    frame4_resized = cv2.resize(frame4, (640, 480))

    top_row = np.hstack((frame1_resized, frame2_resized))
    bottom_row = np.hstack((frame3_resized, frame4_resized))
    combined_frame = np.vstack((top_row, bottom_row))

    cv2.imshow("Combined 4 Videos - YOLO Vehicle Detection", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()
