#include <Arduino.h>
#include <string.h>

// Pin definitions for 4 roads (each road: red, yellow, green)
const int redPins[4]    = {13, 10, 7, 4};
const int yellowPins[4] = {12, 9, 6, 3};
const int greenPins[4]  = {11, 8, 5, 2};

// Fixed yellow light duration (in milliseconds)
const unsigned long yellowTime = 2000; // 2 seconds

// Global counts array for 4 roads; updated continuously from Serial
int counts[4] = {0, 0, 0, 0};
int roadIndex = 0; // Track which road is currently being processed

// Function: Determines green light duration (in seconds) based on vehicle count
int findTime(int cnt) {
  if (cnt >= 50) return 60;
  if (cnt >= 40) return 50;
  if (cnt >= 30) return 40;
  if (cnt >= 20) return 30;
  if (cnt >= 10) return 20;
  return 10;
}

// Helper: Sets all roads to red (turns off yellow and green LEDs)
void setAllRoadsToRed() {
  for (int i = 0; i < 4; i++) {
    digitalWrite(redPins[i], HIGH);
    digitalWrite(yellowPins[i], LOW);
    digitalWrite(greenPins[i], LOW);
  }
}

// Function: Waits for the latest data before processing a road
void waitForLatestData() {
  while (Serial.available() == 0) {
    // Do nothing, just wait
  }
  updateCountsFromSerial();  // Read the latest data
}

// Function: Updates counts from Serial if new data is available
void updateCountsFromSerial() {
  if (Serial.available() > 0) {
    String inputLine = Serial.readStringUntil('\n');
    inputLine.trim(); // Remove extra whitespace

    char buf[50];
    inputLine.toCharArray(buf, 50);

    int index = 0;
    char* token = strtok(buf, " ,"); // Tokenize by space or comma
    while (token != NULL && index < 4) {
      counts[index] = atoi(token);
      token = strtok(NULL, " ,");
      index++;
    }

    // Debugging: Print updated counts
    Serial.print("Updated counts: ");
    for (int i = 0; i < 4; i++) {
      Serial.print(counts[i]);
      Serial.print(" ");
    }
    Serial.println();
  }
}

void setup() {
  // Initialize LED pins as outputs for all roads
  for (int i = 0; i < 4; i++) {
    pinMode(redPins[i], OUTPUT);
    pinMode(yellowPins[i], OUTPUT);
    pinMode(greenPins[i], OUTPUT);
  }
  
  // Start Serial communication
  Serial.begin(9600);
  while (!Serial) {
    ; // Wait for serial port connection (for some boards)
  }
  
  Serial.println("Traffic Light System Initialized.");
  setAllRoadsToRed();
}

void loop() {
  // **Wait for new data before processing the current road**
  waitForLatestData();

  // Get the road to process
  int road = roadIndex;
  int greenSeconds = findTime(counts[road]);
  unsigned long greenTime = greenSeconds * 1000UL; // Convert to milliseconds

  setAllRoadsToRed();  // Ensure all roads are red before turning one green

  // Print the count for the current road
  Serial.print("Processing Road ");
  Serial.print(road + 1);
  Serial.print(" with vehicle count: ");
  Serial.println(counts[road]);

  // Activate the current road (turn green)
  digitalWrite(redPins[road], LOW);
  digitalWrite(greenPins[road], HIGH);

  Serial.print("Road ");
  Serial.print(road + 1);
  Serial.print(" (current count: ");
  Serial.print(counts[road]);
  Serial.print(") green for ");
  Serial.print(greenSeconds);
  Serial.println(" seconds.");

  delay(greenTime);  // Keep green for calculated time

  // Transition to yellow
  digitalWrite(greenPins[road], LOW);
  digitalWrite(yellowPins[road], HIGH);
  Serial.print("Road ");
  Serial.print(road + 1);
  Serial.println(" yellow.");
  delay(yellowTime);

  // Turn yellow off and restore red
  digitalWrite(yellowPins[road], LOW);
  digitalWrite(redPins[road], HIGH);
  
  // **Send acknowledgment after processing one road**
  Serial.println("DONE");

  // Move to the next road (cycle through 0-3)
  roadIndex = (roadIndex + 1) % 4;
}