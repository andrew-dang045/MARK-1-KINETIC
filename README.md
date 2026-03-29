<div align="center">

# MARK I: KINETIC

### *The first presentation controller you wear, not hold.*

**Presenters lose their flow every time they look down. We fixed that.**

[![ESP32](https://img.shields.io/badge/ESP32-WROOM--32E-blue?style=flat-square&logo=espressif)](https://www.espressif.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Modal](https://img.shields.io/badge/Modal-Cloud%20Inference-7C3AED?style=flat-square)](https://modal.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*Built at RevUC Hackathon*

</div>

---

## 👥 The Team

Mark I: Kinetic was conceptualized and built in 24 hours by:

* **Harry Nguyen** – Embedded and Machine Learning Developer.
* **Andrew Dang** – Embedded and System Integration.
* **Huy Le** – CAD and Hardware Prototyping.
* **Chi Nguyen** –  Operations Management and Business Analysis.

---

## What is Kinetic?

Kinetic is a **wearable gesture-control glove** that gives presenters full control of their slides, cursor, and applications — without touching a laptop or holding a clicker.

Draw a letter in the air → the glove recognises it → launches an app or triggers an action. Swipe your wrist → next slide. Hold the button → move the mouse. All wirelessly, from anywhere on stage.

```
Wrist movement  ──► BLE HID  ──► Windows  (mouse + keyboard, instant)
Draw a letter   ──► WiFi TCP ──► Python ──► Modal (cloud AI) ──► action on PC
```

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  ESP32 Glove                    │
│     MPU6050 · KY-024 Hall sensor · 3 buttons    │
└──────────┬──────────────────────┬───────────────┘
           │ BLE HID              │ WiFi TCP
           │ (always on)          │ (on letter gesture)
           ▼                      ▼
    ┌─────────────┐       ┌───────────────┐
    │  Windows PC │       │  Python TCP   │
    │  Mouse +    │       │  Server       │
    │  Keyboard   │       └──────┬────────┘
    └─────────────┘              │ Modal call
                                 ▼
                          ┌─────────────┐
                          │  Modal      │
                          │  Cloud GPU  │──► LSTM Model ──► prediction
                          └──────┬──────┘
                                 │ result + confidence
                                 ▼
                          Action fires on PC (app / browser / command)
```

**Two channels, one glove:**
- **BLE HID** — direct keyboard/mouse link to Windows. Zero latency, no drivers.
- **WiFi TCP** — sensor data streams to Python, which calls Modal for LSTM inference and executes the result locally.

---

## Features

| Feature | How |
|---|---|
| Mouse control | Hold enable button ≥ 300ms → move wrist |
| Left click | Double-click enable button |
| Slide navigation | Hold button + swipe wrist left/right |
| Letter gesture recognition | Hold letter button + draw in air → app launches |
| Pinch-to-zoom | Hold zoom button + move magnet near KY-024 → Ctrl+Scroll |
| Auto drift correction | Gyro offset corrected every 5s when idle |
| Kalman + EMA filtering | Smooth mouse movement, no jitter |

---

## Hardware

| Component | Purpose | Pin |
|---|---|---|
| ESP32-WROOM-32E | Main MCU | — |
| MPU6050 | 6-axis IMU (accel + gyro) | SDA: 22, SCL: 21 |
| KY-024 Linear Hall | Analog zoom sensor | AO: GPIO 34 |
| Enable button | Mouse + swipe control | GPIO 18 |
| Letter button | Gesture recording | GPIO 4 |
| Zoom button | Hall sensor zoom mode | GPIO 19 |

**KY-024 wiring:**
```
KY-024 VCC  →  3.3V
KY-024 GND  →  GND
KY-024 AO   →  GPIO 34  (analog-only pin, 12-bit)
KY-024 DO   →  not used
```

> Turn the blue trimmer on the KY-024 until idle ADC reads ~2048 with no magnet nearby.

---

## Project Structure

```
MARK-1-KINETIC/
│
├── Firmware/
│   ├── main.cpp                      # Orchestrates all modules
│   ├── MouseControl/
│   │   ├── mouse-control.h
│   │   └── mouse-control.cpp         # BLE HID mouse, buttons, letter recording
│   ├── GestureControl/
│   │   ├── gesture-control.h
│   │   └── gesture-control.cpp       # Swipe detection → arrow keys
│   ├── WiFiTcp/
│   │   ├── wifi-tcp.h
│   │   └── wifi-tcp.cpp              # WiFi + TCP streaming to Python
│   └── HallZoom/
│       ├── hall-zoom.h
│       └── hall-zoom.cpp             # KY-024 → Ctrl+Scroll zoom
│
├── PythonML/
│   ├── collect_training_data.py      # Data collection + live 6-axis waveform
│   ├── train_model.py                # Bidirectional LSTM training (NVIDIA Ada)
│   ├── modal_inference.py            # Modal cloud deployment
│   └── glove_server.py               # TCP server → Modal → action
│
└── Tests/
    └── hall_zoom_test.ino            # Standalone KY-024 test sketch
```

---

## Setup

### 1. Firmware

Install Arduino libraries:
```
BleCombo       (Library Manager)
Wire           (built-in)
WiFi           (built-in)
```

Edit `WiFiTcp/wifi-tcp.h` before flashing:
```cpp
#define WIFI_SSID     "YourHotspotName"
#define WIFI_PASSWORD "YourPassword"
#define PC_IP         "192.168.X.X"   // run ipconfig on Windows to find this
#define PC_PORT       9000
```

Flash `main.cpp` → open Serial Monitor at `115200` baud → keep hand **completely still** during the 4-second calibration.

### 2. Python dependencies

```bash
pip install pyserial numpy matplotlib tensorflow scikit-learn scipy modal bleak
```

### 3. Collect training data

```bash
python collect_training_data.py
```

Connect the ESP32 via **USB** for data collection. Watch the live 6-axis waveform:
- Flat lines = hand not moving → auto-rejected
- Clear waves = good gesture → keep it

```python
# configure at top of script
SERIAL_PORT          = 'COM10'
RECORDING_DURATION_S = 2.2       # must match LETTER_RECORDING_DURATION in firmware
letters              = ['C', 'P', 'I']
samples_per          = 7 (More Samples More Accurate)
```

### 4. Train the model

```bash
python train_model.py
```

Outputs three files needed at inference time:
```
letter_recognition_model.keras
feature_norm.npy                 # normalisation — required at inference
label_encoder.pkl                # index → letter mapping
```

### 5. Deploy to Modal

```bash
modal deploy modal_inference.py
```

### 6. Run the glove server

```bash
python glove_server.py
```

The server prints your local IP automatically. Set `PC_IP` in `wifi-tcp.h` to that address.

Add or edit letter → action mappings:
```python
ACTIONS = {
    'C': lambda: webbrowser.open("http://google.com"),
    'P': lambda: os.system("start powerpnt"),
    'I': lambda: os.system("start outlook"),
}
```

---

## Button Reference

| Action | Result |
|---|---|
| Hold enable (GPIO 18) ≥ 200ms | Mouse movement active |
| Double-click enable (each press < 180ms) | Left mouse button |
| Hold enable + swipe wrist | Left / Right arrow key |
| Hold letter button (GPIO 4) | 2-second gesture recording |
| Hold zoom button (GPIO 19) + move magnet | Ctrl + Scroll zoom |

---

## Serial Tuning Commands

While connected via USB Serial at 115200:

| Command | Effect |
|---|---|
| `x0.4` | Set X mouse sensitivity to 0.4 |
| `y0.3` | Set Y mouse sensitivity to 0.3 |
| `d3.0` | Set gyro dead zone to 3.0 |
| `r` | Recalibrate gyro offsets |
| `s` | Print all current settings |

---

## Gesture Tips

The MPU6050 measures **angular velocity** — how fast your wrist rotates — not pen position. Think conducting an orchestra, not writing on paper.

| Letter | Suggested wrist motion |
|---|---|
| `C` | Wide outward arc, then back |
| `P` | Sharp downward flick, then outward bump |
| `I` | Down sharply and stay at one point |

Consistency matters more than the specific motion — train and use the same gesture every time.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| GPIO 2 LED flickering | Move letter button wire to GPIO 4 — GPIO 2 has onboard LED |
| Always predicts same letter | Duration mismatch — check `RECORDING_DURATION_S` matches ESP32 `LETTER_RECORDING_DURATION` |
| Low confidence predictions | Collect more samples; ensure consistent gesture motion |
| MPU6050 FAILED on boot | Check SDA=22, SCL=21 wiring; confirm 3.3V power |
| Hall sensor stuck at 0 or 4095 | Adjust blue trimmer; check 3.3V supply |
| BLE not connecting to Windows | Unpair from Bluetooth settings and re-pair |

---

## ML Model Details

- **Architecture:** Bidirectional LSTM + Multi-Head Self-Attention + residual dense block
- **Input:** 150 × 6 (samples × axes: ax, ay, az, gx, gy, gz)
- **Training:** 5-fold stratified cross-validation, label smoothing, class balancing
- **Augmentation:** jitter, scaling, time-warp, rotation (4× dataset)
- **Precision:** Mixed FP16 (optimised for NVIDIA Ada tensor cores)
- **Inference:** Modal cloud GPU (T4), confidence threshold gate at 65%

---

## Credits & Acknowledgments

This project stands on the shoulders of several open-source implementations and algorithmic foundations:

* **BLE Mouse Implementation:** Core Bluetooth HID logic and mouse movement structures were adapted from [takeyamayuki's GyroMouseBLE](https://github.com/takeyamayuki/GyroMouseBLE).
* **Signal Processing:**
    * **Kalman Filter:** Implemented based on the [KalmanFilter library](https://github.com/denyssene/SimpleKalmanFilter) for smoothing MPU6050 accelerometer and gyroscope data.
    * **EMA Filter:** Exponential Moving Average logic used for low-latency jitter reduction in mouse cursor tracking.
* **Machine Learning:** Architecture inspired by common LSTM-based gesture recognition patterns used in wearable IMU research. (https://archive.ics.uci.edu/dataset/59/letter+recognition)
* **Special Thanks:** Built during **RevolutionUC 2026 Hackathon** at the 1819 Innovation Hub.

<div align="center">

*"Designed for the few. Built for everyone."*

**MARK I: KINETIC — RevolutionUC 2026 Hackathon**

</div>
