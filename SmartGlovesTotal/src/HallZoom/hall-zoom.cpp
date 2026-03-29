// ===========================================
// HALL ZOOM MODULE — Implementation
//
// KY-024 AO on GPIO 34 → analogRead (0–4095, 12-bit)
// Zoom button on GPIO 19 → INPUT_PULLUP
//
// How it works:
//   1. Button pressed → capture baseline ADC reading
//   2. Every loop, read ADC → compute delta from baseline
//   3. Map delta → scroll amount (with dead zone)
//   4. Hold Ctrl via BLE HID + send scroll → Ctrl+Scroll zoom
//   5. Button released → release Ctrl, clear state
//
// ADC note:
//   ESP32 ADC is non-linear near 0 and 4095.
//   We clamp to 200–3900 to stay in the linear region.
//   KY-024 at rest (no magnet) ≈ 2048 (mid-rail).
//   Trimmer adjusts gain — turn it until resting value
//   is close to 2048 with no magnet nearby.
// ===========================================

#include "hall-zoom.h"
#include <BleCombo.h>
#include <Arduino.h>

// ── Pin config ────────────────────────────────────────────────────────────────
#define HALL_PIN     34   // AO — analog only GPIO, 12-bit
#define ZOOM_BTN_PIN 19   // zoom enable button

// ── Tuning ────────────────────────────────────────────────────────────────────
// Dead zone: ADC units either side of baseline that are ignored.
// Prevents jitter when holding still. Increase if noisy.
static const int   DEAD_ZONE_ADC    = 80;

// Sensitivity: ADC units of delta needed to fire one scroll tick.
// Lower = more sensitive. 60 is a good starting point.
static const int   UNITS_PER_TICK   = 60;

// Max scroll ticks per call (caps runaway zoom if magnet moves fast)
static const int   MAX_TICKS        = 5;

// How many ms between scroll events (throttle rate)
static const unsigned long SCROLL_INTERVAL_MS = 60;

// ADC linear region clamp
static const int   ADC_MIN          = 200;
static const int   ADC_MAX          = 3900;

// Button debounce
static const unsigned long DEBOUNCE_MS = 40;

// ── State ─────────────────────────────────────────────────────────────────────
static bool          zoomActive      = false;
static int           baseline        = 2048;   // recaptured on each button press
static bool          ctrlHeld        = false;

static bool          btnLastState    = HIGH;
static bool          btnState        = HIGH;
static unsigned long btnDebounceTime = 0;
static bool          btnPressed      = false;

static unsigned long lastScrollTime  = 0;

// EMA smoothing for ADC (reduces noise)
static float         emaValue        = 2048.0f;
static const float   EMA_ALPHA       = 0.25f;   // lower = smoother, more lag

// ── Helpers ───────────────────────────────────────────────────────────────────
static int readHallSmoothed() {
  int raw = analogRead(HALL_PIN);
  raw = constrain(raw, ADC_MIN, ADC_MAX);
  emaValue = EMA_ALPHA * raw + (1.0f - EMA_ALPHA) * emaValue;
  return (int)emaValue;
}

static void updateButton() {
  int reading = digitalRead(ZOOM_BTN_PIN);
  unsigned long ct = millis();

  if (reading != btnLastState) btnDebounceTime = ct;

  if ((ct - btnDebounceTime) > DEBOUNCE_MS) {
    if (reading != btnState) {
      btnState = reading;
      btnPressed = (btnState == LOW);
    }
  }
  btnLastState = reading;
}

// ── Public API ────────────────────────────────────────────────────────────────
void setupHallZoom() {
  pinMode(ZOOM_BTN_PIN, INPUT_PULLUP);
  pinMode(HALL_PIN,     INPUT);          // GPIO 34 is input-only

  // Set ADC resolution and attenuation for full 0–3.3V range
  analogReadResolution(12);             // 12-bit → 0–4095
  analogSetAttenuation(ADC_11db);       // full 3.3V range

  // Warm up EMA with a few reads
  for (int i = 0; i < 20; i++) {
    readHallSmoothed();
    delay(5);
  }
  baseline = (int)emaValue;

  Serial.println("\n╔═══════════════════════════════════════════╗");
  Serial.println("║        Hall Zoom Module Ready             ║");
  Serial.println("╠═══════════════════════════════════════════╣");
  Serial.printf( "║  Hall pin    : GPIO %2d (AO, 12-bit)       ║\n", HALL_PIN);
  Serial.printf( "║  Button pin  : GPIO %2d                    ║\n", ZOOM_BTN_PIN);
  Serial.printf( "║  Baseline    : %4d ADC counts             ║\n", baseline);
  Serial.printf( "║  Dead zone   : ±%3d ADC counts            ║\n", DEAD_ZONE_ADC);
  Serial.printf( "║  Sensitivity : %3d counts / scroll tick   ║\n", UNITS_PER_TICK);
  Serial.println("║                                           ║");
  Serial.println("║  Hold button + move magnet to zoom        ║");
  Serial.println("║  Closer = zoom in   Away = zoom out       ║");
  Serial.println("╚═══════════════════════════════════════════╝\n");
}

void loopHallZoom() {
  updateButton();
  unsigned long ct = millis();

  // ── Button just pressed → enter zoom mode, recapture baseline ────────────
  if (btnPressed && !zoomActive) {
    zoomActive = true;

    // Recapture baseline at moment of press so drift doesn't matter
    // Flush EMA first with a quick burst of reads
    for (int i = 0; i < 10; i++) { readHallSmoothed(); delay(2); }
    baseline = (int)emaValue;

    // Hold Ctrl via BLE HID
    Keyboard.press(KEY_LEFT_CTRL);
    ctrlHeld = true;

    Serial.printf("[Zoom] Active  baseline=%d\n", baseline);
  }

  // ── Button released → exit zoom mode ─────────────────────────────────────
  if (!btnPressed && zoomActive) {
    zoomActive = false;

    if (ctrlHeld) {
      Keyboard.release(KEY_LEFT_CTRL);
      ctrlHeld = false;
    }

    Serial.println("[Zoom] Inactive");
    return;
  }

  // ── Zoom active → read sensor and scroll ─────────────────────────────────
  if (!zoomActive) return;
  if (ct - lastScrollTime < SCROLL_INTERVAL_MS) return;   // throttle

  int current = readHallSmoothed();
  int delta   = current - baseline;   // positive = magnet closer (field stronger)

  // Apply dead zone
  if (abs(delta) < DEAD_ZONE_ADC) return;

  // Remove dead zone from magnitude so scroll starts at 0 (no jump)
  int adjusted = (delta > 0)
    ? (delta - DEAD_ZONE_ADC)
    : (delta + DEAD_ZONE_ADC);

  // Convert to scroll ticks
  int ticks = adjusted / UNITS_PER_TICK;
  ticks = constrain(ticks, -MAX_TICKS, MAX_TICKS);

  if (ticks == 0) return;

  // Ctrl is already held — send scroll wheel
  // Positive ticks = scroll up = zoom in
  // Negative ticks = scroll down = zoom out
  Mouse.move(0, 0, ticks);

  lastScrollTime = ct;

  Serial.printf("[Zoom] ADC=%4d  delta=%+4d  ticks=%+d\n", current, delta, ticks);
}

bool isZoomActive() { return zoomActive; }