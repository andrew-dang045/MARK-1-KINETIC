#include "GestureControl/gesture-control.h"
#include "MouseControl/mouse-control.h"
#include <Arduino.h>


// If your WiFi module requires initialization in setup(), include its header
// here #include "WiFiTcp/wifi-tcp.h"

void setup() {
  // 1. Initialize Serial communication
  Serial.begin(115200);

  // 2. Initialize WiFi/TCP if your wifi-tcp.h module requires a setup function
  // setupWiFi();

  // 3. Initialize the Accessibility Glove modules
  // setupMouse() initializes the MPU6050, configures buttons, and calibrates
  // offsets. setupGestures() initializes the Keyboard BLE interface.
  setupMouse();
  setupGestures();
}

void loop() {
  // 1. Run the core mouse, sensor, and letter recognition logic
  loopMouse();

  // 2. Run the gesture recognition logic (swipes)
  loopGestures();

  // 3. Optional: Add a hook to tune gesture settings via Serial
  // (mouse-control.cpp already handles some Serial commands, you can add
  // gesture ones here or move them)
  if (Serial.available()) {
    char cmd = Serial.peek(); // Just peek to see if it's a gesture command
    if (cmd == 'g') {
      Serial.read(); // consume 'g'
      printGestureSettings();
    } else if (cmd == 't') {
      Serial.read(); // consume 't'
      float newThresh = Serial.parseFloat();
      setSwipeThreshold(newThresh);
    } else if (cmd == 'c') {
      Serial.read(); // consume 'c'
      unsigned long newCooldown = Serial.parseInt();
      setSwipeCooldown(newCooldown);
    }
    // Note: Other commands (x, y, d, r, s) are handled inside loopMouse()
  }
}