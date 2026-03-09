import tkinter as tk
import serial
import time
import sys
from tkinter import font as tkFont

# --- SERIAL CONFIGURATION ---
ARDUINO_PORT = "COM6"  # Change as needed
BAUD_RATE = 9600

arduino = None

# Attempt connection to Arduino
try:
    arduino = serial.Serial(port=ARDUINO_PORT, baudrate=BAUD_RATE, timeout=0.1)
    print(f"Connected to Arduino on {ARDUINO_PORT}")
    time.sleep(2)  # Wait for Arduino reset
except serial.SerialException as e:
    print(f"ERROR: Could not connect to Arduino on {ARDUINO_PORT}: {e}")
    arduino = None
except Exception as e:
    print(f"Unexpected error: {e}")
    arduino = None


def send_command(command_char):
    """Send a single character command to the Arduino."""
    if arduino and arduino.is_open:
        try:
            arduino.write(bytes(command_char, 'utf-8'))
            arduino.flush()
            print(f"Sent command: '{command_char}'")
        except Exception as e:
            print(f"Error sending command '{command_char}': {e}")
    else:
        print(f"Arduino not connected. Cannot send command '{command_char}'.")


# Tkinter GUI setup
root = tk.Tk()
root.title("Traffic Scenario Controller")
root.geometry("360x420")
root.configure(bg='whitesmoke')

title_font = tkFont.Font(family="Helvetica", size=16, weight="bold")
button_font = tkFont.Font(family="Helvetica", size=12)

title_label = tk.Label(root, text="Select Traffic Scenario", font=title_font, bg='whitesmoke', fg='black')
title_label.pack(pady=20)

button_style = {
    "width": 28,
    "font": button_font,
    "bg": "white",
    "fg": "black",
    "relief": tk.FLAT,
    "bd": 1,
    "highlightbackground": "gray"
}

# Scenario buttons
tk.Button(root, text="Scenario 1: Normal Flow", command=lambda: send_command('1'), **button_style).pack(pady=10, ipady=8)
tk.Button(root, text="Scenario 2: Rush Hour (Priority 2 & 4)", command=lambda: send_command('2'), **button_style).pack(pady=10, ipady=8)
tk.Button(root, text="Scenario 3: Night Time (Fast Cycle)", command=lambda: send_command('3'), **button_style).pack(pady=10, ipady=8)

# STOP button (All Red)
stop_bg = "#CC0000"
stop_active = "#FF3333"
tk.Button(root, text="STOP (All Red)", command=lambda: send_command('4'),
          width=28, font=button_font,
          bg=stop_bg, fg="white",
          relief=tk.FLAT,
          activebackground=stop_active,
          activeforeground="white").pack(pady=25, ipady=10)


def on_close():
    print("Closing application...")
    if arduino and arduino.is_open:
        try:
            # Send STOP command (all red) before closing
            arduino.write(b'4')
            arduino.flush()
            print("Sent STOP signal before close.")
            time.sleep(0.1)
            arduino.close()
            print("Serial port closed cleanly.")
        except Exception as e:
            print(f"Error during closing serial port: {e}")
    root.destroy()
    sys.exit(0)


root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()
