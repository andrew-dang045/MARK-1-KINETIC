// ===========================================
// GESTURE CONTROL MODULE
// Detects wrist swipe gestures using gyro Z
// and sends arrow key presses over BLE HID.
//
// SWIPE AXIS:
//   Gyro Z (dpsZ) measures wrist rotation
//   around the vertical axis (like turning a
//   door knob). A fast rotation left or right
//   is what triggers a swipe.
//
// ACTIVE CONDITION:
//   Swipe only fires while enable button IS
//   held (isMovementEnabled() == true).
//   Releasing the button disables swipe.
// ===========================================

#include "gesture-control.h"
#include "MouseControl/mouse-control.h"
#include <BleCombo.h>
#include <Arduino.h>

// ===========================================
// TUNING PARAMETERS
// ===========================================

// SWIPE_THRESHOLD (°/s):
//   Minimum gyro Z speed to register as a swipe.
//   ↑ Higher = harder to trigger, needs faster flick
//   ↓ Lower  = easier to trigger, may false-fire
//   Typical range: 200 (sensitive) – 600 (firm)
float SWIPE_THRESHOLD = 150;

// SWIPE_COOLDOWN (ms):
//   Minimum time between two swipe events.
//   Prevents one fast flick from firing twice.
//   ↑ Higher = slower repeat rate
//   ↓ Lower  = can swipe again sooner
unsigned long SWIPE_COOLDOWN = 800;

// ── Internal state ────────────────────────────────────────────────────────────
unsigned long lastSwipeTime   = 0;     // timestamp of last confirmed swipe
bool          swipeInProgress = false; // true while wrist is mid-swipe

// Direction result type
enum SwipeDirection {
  NO_SWIPE,
  SWIPE_LEFT,
  SWIPE_RIGHT
};

// ===========================================
// SWIPE DETECTION
//
// How it works:
//   1. Read corrected gyro Z (raw minus offset)
//   2. If |corrected_z| > SWIPE_THRESHOLD AND
//      cooldown has elapsed AND no swipe already
//      in progress → fire the swipe direction
//   3. Mark swipeInProgress = true to prevent
//      re-triggering while wrist is still moving
//   4. Reset swipeInProgress once gyro Z drops
//      back below SWIPE_THRESHOLD / 2 (hysteresis)
//      so the next swipe can register cleanly
//
// Hysteresis (SWIPE_THRESHOLD / 2 = 225):
//   The reset threshold is intentionally lower
//   than the trigger threshold. This means the
//   wrist must return close to neutral before
//   another swipe can fire — prevents bouncing.
// ===========================================
SwipeDirection detectSwipe(float raw_z, float offset_z) {
  unsigned long ct = millis();

  // ── Cooldown gate ─────────────────────────────────────────────────────────
  // Block all swipes until SWIPE_COOLDOWN ms have passed since the last one
  if (ct - lastSwipeTime < SWIPE_COOLDOWN) return NO_SWIPE;

  float corrected_z = raw_z - offset_z;

  // ── Trigger zone: |corrected_z| > SWIPE_THRESHOLD ────────────────────────
  if (abs(corrected_z) > SWIPE_THRESHOLD) {
    if (!swipeInProgress) {
      swipeInProgress = true;
      lastSwipeTime   = ct;

      // Positive Z = wrist rotates one way  → SWIPE RIGHT → Right Arrow
      // Negative Z = wrist rotates other way → SWIPE LEFT  → Left Arrow
      if (corrected_z > SWIPE_THRESHOLD)  return SWIPE_RIGHT;
      if (corrected_z < -SWIPE_THRESHOLD) return SWIPE_LEFT;
    }

  // ── Reset zone: |corrected_z| < SWIPE_THRESHOLD / 2 (hysteresis) ─────────
  } else {
    if (abs(corrected_z) < SWIPE_THRESHOLD / 2.0f) {
      swipeInProgress = false;  // wrist back at rest, ready for next swipe
    }
  }

  return NO_SWIPE;
}

// ===========================================
// SWIPE HANDLER
//
// Called every loop. Checks the active condition
// then calls detectSwipe and sends the BLE key.
//
// Active condition:
//   isMovementEnabled() == true
//   i.e. the enable button must be HELD.
//   Swipe is intentionally disabled when button
//   is not held to avoid accidental triggers.
// ===========================================
void handleSwipeGestures(float raw_z, float offset_z) {
  if (!isMovementEnabled()) return;   // swipe only works while button held

  SwipeDirection swipe = detectSwipe(raw_z, offset_z);

  if (swipe == SWIPE_RIGHT) {
    Serial.println("[Gesture] SWIPE RIGHT → Right Arrow");
    Keyboard.press(KEY_RIGHT_ARROW);
    Keyboard.releaseAll();
  } else if (swipe == SWIPE_LEFT) {
    Serial.println("[Gesture] SWIPE LEFT → Left Arrow");
    Keyboard.press(KEY_LEFT_ARROW);
    Keyboard.releaseAll();
  }
}

// ===========================================
// SETUP
// ===========================================
void setupGestures() {
  Keyboard.begin();

  Serial.println("\n╔═══════════════════════════════════════════╗");
  Serial.println("║     Accessibility Glove - Gesture Module  ║");
  Serial.println("╠═══════════════════════════════════════════╣");
  Serial.printf( "║  Swipe threshold : %4.0f °/s               ║\n", SWIPE_THRESHOLD);
  Serial.printf( "║  Swipe cooldown  : %4lums                 ║\n",  SWIPE_COOLDOWN);
  Serial.println("║                                           ║");
  Serial.println("║  Hold enable btn + swipe wrist:           ║");
  Serial.println("║  • Swipe RIGHT  → Right Arrow Key         ║");
  Serial.println("║  • Swipe LEFT   → Left Arrow Key          ║");
  Serial.println("╚═══════════════════════════════════════════╝\n");
}

// ===========================================
// MAIN LOOP
//
// offset_z is static at 0 because the gyro
// calibration offsets are already applied
// inside mouse-control.cpp before getRawGyroZ()
// returns the value. No double-correction needed.
// ===========================================
void loopGestures() {
  float raw_z      = getRawGyroZ();
  static float offset_z = 0;          // offset already baked in by mouse-control
  handleSwipeGestures(raw_z, offset_z);
}

// ===========================================
// RUNTIME CONFIGURATION
// Callable from Serial or other modules to
// tune behaviour without reflashing.
// ===========================================

// Raise threshold → harder swipe needed
// Lower threshold → easier swipe, more sensitive
void setSwipeThreshold(float threshold) {
  SWIPE_THRESHOLD = threshold;
  Serial.print("[Gesture] Threshold → "); Serial.println(SWIPE_THRESHOLD);
}

// Raise cooldown → longer wait between swipes
// Lower cooldown → can swipe again sooner
void setSwipeCooldown(unsigned long cooldown) {
  SWIPE_COOLDOWN = cooldown;
  Serial.print("[Gesture] Cooldown → "); Serial.println(SWIPE_COOLDOWN);
}

void printGestureSettings() {
  Serial.println("\n═══ GESTURE SETTINGS ═══════════════");
  Serial.printf("  Threshold : %.0f °/s  (higher = harder)\n", SWIPE_THRESHOLD);
  Serial.printf("  Cooldown  : %lums    (higher = slower repeat)\n", SWIPE_COOLDOWN);
  Serial.printf("  Active    : %s\n", isMovementEnabled() ? "YES (btn held)" : "NO (btn released)");
  Serial.println("════════════════════════════════════\n");
}