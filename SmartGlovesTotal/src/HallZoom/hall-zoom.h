// ===========================================
// HALL ZOOM MODULE
// KY-024 Linear Hall Effect Sensor → Zoom
//
// Wiring:
//   KY-024 AO  → GPIO 34  (analog, 12-bit)
//   KY-024 VCC → 3.3V
//   KY-024 GND → GND
//   (DO pin unused — we use analog for resolution)
//
//   Zoom button → GPIO 19  (INPUT_PULLUP)
//
// Behaviour:
//   Hold zoom button → zoom mode active
//   Move magnet closer  → Ctrl + Scroll Up   (zoom in)
//   Move magnet away    → Ctrl + Scroll Down  (zoom out)
//   Release button      → zoom mode off, Ctrl released
//
// Calibration:
//   On first hold, the sensor reading is captured as the
//   neutral baseline. Deviation from baseline drives zoom.
//   Blue trimmer on KY-024 adjusts sensitivity range.
// ===========================================

#ifndef HALL_ZOOM_H
#define HALL_ZOOM_H

#include <Arduino.h>

// Call once in setup()
void setupHallZoom();

// Call every loop() — handles reading, button, and BLE HID zoom
void loopHallZoom();

// Returns true while zoom mode is active (button held)
bool isZoomActive();

#endif