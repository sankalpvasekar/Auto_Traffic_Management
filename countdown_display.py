import serial
import time
import os
from datetime import datetime

class TrafficMonitor:
    def __init__(self, port, filename):
        self.ser = serial.Serial(port, 9600, timeout=0.1)  # Reduced timeout
        self.filename = filename
        self.status = {1: {'state': '🔴', 'countdown': 0},
                     2: {'state': '🔴', 'countdown': 0},
                     3: {'state': '🔴', 'countdown': 0},
                     4: {'state': '🔴', 'countdown': 0}}
        self.current_road = None
        time.sleep(2)
    
    def display_status(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print(f"=== Real-Time Traffic Status ({datetime.now().strftime('%H:%M:%S')}) ===")
        for road, data in self.status.items():
            print(f"Road {road}: {data['state']} {data['countdown']}s")
        print("======================================")
    
    def process_serial(self):
        while self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode().strip()
                if line.startswith("COUNTDOWN"):
                    parts = line.split()
                    if len(parts) == 3:
                        road = int(parts[1])
                        seconds = int(parts[2])
                        self.current_road = road
                        self.status[road]['state'] = '🟢'
                        self.status[road]['countdown'] = seconds
                        self.display_status()
                        
                        # Automatically revert to red when countdown ends
                        if seconds == 1:
                            self.status[road]['state'] = '🔴'
                            self.display_status()
                            
                    return False
                elif line == "DONE":
                    if self.current_road:
                        self.status[self.current_road]['state'] = '🔴'
                        self.status[self.current_road]['countdown'] = 0
                        self.display_status()
                    return True
            except UnicodeDecodeError:
                continue
        return False
    
    def run(self):
        try:
            while True:
                try:
                    with open(self.filename, 'r') as f:
                        data = f.read().strip()
                    
                    if data and self.validate_data(data):
                        print(f"\nSending new counts: {data}")
                        self.ser.reset_input_buffer()
                        self.ser.write(f"{data}\n".encode('utf-8'))
                        
                        start_time = time.time()
                        while True:
                            if self.process_serial():
                                print("Cycle completed. Ready for new data.")
                                break
                            
                            if time.time() - start_time > 60:
                                print("Warning: Cycle taking longer than expected")
                                break
                            
                            time.sleep(0.05)  # More responsive polling
                    
                    time.sleep(0.5)  # Reduced file check interval
                
                except FileNotFoundError:
                    print("Data file missing!")
                    time.sleep(2)
                
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.ser.close()

    def validate_data(self, data):
        try:
            counts = list(map(int, data.split()))
            return len(counts) == 4 and all(c >= 0 for c in counts)
        except ValueError:
            print("Invalid data format in file")
            return False

if __name__ == "__main__":
    monitor = TrafficMonitor('/dev/ttyACM0', 'carsCount.txt')
    monitor.run()