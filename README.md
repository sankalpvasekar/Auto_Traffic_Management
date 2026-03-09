# Smart Traffic Signal System

## 🎥 YouTube Demo
(Add YouTube video link here after upload)

## Project Description
An intelligent traffic management system that uses computer vision and YOLOv8 for real-time vehicle detection to dynamically control traffic signals. The system processes multiple video feeds simultaneously, counts vehicles in designated regions of interest, and optimizes traffic flow by adjusting signal timing based on current traffic conditions.

## Tech Stack
- **Python 3.x**
- **OpenCV** - Video processing and computer vision
- **YOLOv8 (Ultralytics)** - Real-time vehicle detection
- **NumPy** - Numerical computations
- **PySerial** - Arduino communication
- **Tkinter** - GUI for traffic scenario control
- **Arduino** - Hardware traffic signal controller

## How to Run
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Connect Arduino:**
   - Connect Arduino to COM6 (or modify port in code)
   - Ensure traffic signal hardware is properly connected

3. **Run the main traffic system:**
   ```bash
   python traffic2.py
   ```

4. **Alternative interfaces:**
   - GUI Control: `python gui.py`
   - Multi-video processing: `python multi_video_processing.py`
   - Countdown display: `python countdown_display.py`

## Local Server
This system runs locally and connects to Arduino hardware for traffic signal control. No web server required.

## Features
- Real-time vehicle detection using YOLOv8
- Multi-camera video processing
- Dynamic traffic signal timing based on vehicle count
- Three traffic scenarios: Normal, Rush Hour, Night Time
- ROI-based vehicle counting for accuracy
- Arduino integration for hardware control
- GUI for manual scenario selection
