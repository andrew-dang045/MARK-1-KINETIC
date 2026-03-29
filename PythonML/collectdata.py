import serial
import time
import json
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# ── Serial config ──────────────────────────────────────────────────────────────
SERIAL_PORT = 'COM10'  # Double check your COM port!
BAUD_RATE = 115200
WINDOW_SIZE = 200

# ── Recording duration — must match LETTER_RECORDING_DURATION in mouse-control.cpp ──
RECORDING_DURATION_S = 2.2  # ESP32 streams 3000 ms + 0.2 s latency buffer

# ── Shared state ───────────────────────────────────────────────────────────────
lock = threading.Lock()
buf_ax = deque(maxlen=WINDOW_SIZE)
buf_ay = deque(maxlen=WINDOW_SIZE)
buf_az = deque(maxlen=WINDOW_SIZE)
buf_gx = deque(maxlen=WINDOW_SIZE)
buf_gy = deque(maxlen=WINDOW_SIZE)
buf_gz = deque(maxlen=WINDOW_SIZE)

recording = []
is_recording = False
stop_reader = False

SKIP_TOKENS = ('ets', 'rst:', 'boot:', '╔', '║', '═', 'Waiting', 'MPU6050')


# ══════════════════════════════════════════════════════════════════════════════
# Background serial reader
# ══════════════════════════════════════════════════════════════════════════════
def serial_reader(ser):
    global is_recording, stop_reader
    while not stop_reader:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8', errors='replace').strip()
                if not line or any(t in line for t in SKIP_TOKENS):
                    continue
                values = [float(x) for x in line.split(',')]
                if len(values) != 6:
                    continue
                ax_, ay_, az_, gx_, gy_, gz_ = values
                with lock:
                    buf_ax.append(ax_)
                    buf_ay.append(ay_)
                    buf_az.append(az_)
                    buf_gx.append(gx_)
                    buf_gy.append(gy_)
                    buf_gz.append(gz_)
                    if is_recording:
                        recording.append(values)
            except ValueError:
                pass
            except Exception as e:
                print(f"❌ Reader error: {e}")
        else:
            time.sleep(0.001)


# ══════════════════════════════════════════════════════════════════════════════
# Real-time Gx vs Gy Dot Trail Figure
# ══════════════════════════════════════════════════════════════════════════════
def make_figure():
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0d0d0d')
    fig.suptitle('IMU Live Trail (Gx vs Gy)', color='#e0e0e0', fontsize=14,
                 fontfamily='monospace', y=0.95)

    ax.set_facecolor('#111111')
    ax.tick_params(colors='#666')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    ax.set_xlabel('Gyro X (°/s)', color='#aaa', fontsize=10)
    ax.set_ylabel('Gyro Y (°/s)', color='#aaa', fontsize=10)

    AXIS_PAD = 20.0
    # FIX: removed set_aspect('equal', adjustable='datalim') — it was fighting
    # the dynamic xlim/ylim updates and collapsing the y-axis to a tiny range.
    # X and Y now scale independently so both axes always fill the canvas.
    ax.set_xlim(-AXIS_PAD, AXIS_PAD)
    ax.set_ylim(-AXIS_PAD, AXIS_PAD)

    ax.axhline(0, color='#333', lw=1, zorder=1)
    ax.axvline(0, color='#333', lw=1, zorder=1)

    trail_line, = ax.plot([], [], color='white', alpha=0.3, lw=1.5, zorder=2)
    trail_scatter = ax.scatter([], [], c=[], cmap='viridis', s=40, zorder=3, vmin=0, vmax=1)

    rec_text = ax.text(
        0.03, 0.94, '', transform=ax.transAxes,
        color='#ff3030', fontsize=12, fontfamily='monospace', fontweight='bold'
    )
    status_text = ax.text(
        0.03, 0.03, 'Samples: 0', transform=ax.transAxes,
        color='#555', fontsize=9, fontfamily='monospace'
    )

    def update(_frame):
        with lock:
            gx = list(buf_gx)
            gy = list(buf_gy)
            n = len(recording)
            rec = is_recording

        if not gx:
            return trail_line, trail_scatter, rec_text, status_text

        trail_line.set_data(gx, gy)

        offsets = np.c_[gx, gy]
        trail_scatter.set_offsets(offsets)

        colors = np.linspace(0, 1, len(gx))
        trail_scatter.set_array(colors)

        # Scale X and Y independently — no set_aspect conflict
        gx_arr = np.array(gx)
        gy_arr = np.array(gy)
        x_margin = max(AXIS_PAD, abs(gx_arr).max() * 1.15)
        y_margin = max(AXIS_PAD, abs(gy_arr).max() * 1.15)
        ax.set_xlim(-x_margin, x_margin)
        ax.set_ylim(-y_margin, y_margin)

        rec_text.set_text('⏺ RECORDING' if rec else '')
        status_text.set_text(f'Captured for JSON: {n}')

        return trail_line, trail_scatter, rec_text, status_text

    ani = animation.FuncAnimation(
        fig, update, interval=33,
        blit=False,
        cache_frame_data=False
    )
    return fig, ani


# ══════════════════════════════════════════════════════════════════════════════
# Main collection loop
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global is_recording, stop_reader, recording

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()

    reader_thread = threading.Thread(target=serial_reader, args=(ser,), daemon=True)
    reader_thread.start()

    fig, ani = make_figure()
    plt.show(block=False)
    plt.pause(0.1)

    # ── Collection UI ──────────────────────────────────────────────────────
    letters = ['C', 'P', 'I']
    samples_per = 7
    training_data = []  # list of {'letter': ..., 'sensor_data': ...}

    print("\n╔═══════════════════════════════════════════╗")
    print("║       TRAINING DATA COLLECTION            ║")
    print("╠═══════════════════════════════════════════╣")
    print("║  After each capture:                      ║")
    print("║    Enter → keep & continue                ║")
    print("║    r     → redo this sample               ║")
    print("║    u     → undo last accepted sample      ║")
    print("║    s     → skip to next letter            ║")
    print("╚═══════════════════════════════════════════╝")
    print(f"\n  Recording window : {RECORDING_DURATION_S:.1f} s\n")

    for letter in letters:
        print(f"\n{'=' * 50}")
        print(f"  Collecting data for letter: [{letter}]")
        print(f"{'=' * 50}")

        i = 0
        while i < samples_per:
            input(f"\n[{i + 1}/{samples_per}] Press Enter → HOLD button → draw '{letter}'...")

            with lock:
                recording = []
                buf_gx.clear()
                buf_gy.clear()
                is_recording = True

            print(f"⏺  Recording {RECORDING_DURATION_S:.1f} s… GO!")
            fig.suptitle(f"⏺  RECORDING  '{letter}'  [{i + 1}/{samples_per}]",
                         color='#ff4040', fontsize=13, fontfamily='monospace')
            plt.pause(RECORDING_DURATION_S)

            with lock:
                is_recording = False
                captured = list(recording)

            fig.suptitle('IMU Live Trail (Gx vs Gy)', color='#e0e0e0',
                         fontsize=14, fontfamily='monospace')
            plt.pause(0.05)

            # Auto-retry if too few samples — no point asking
            if len(captured) <= 10:
                print(f"❌ Only {len(captured)} samples — too few, auto-retrying...")
                continue

            # ── Ask what to do with this capture ──────────────────────────
            print(f"✅ Captured {len(captured)} samples")
            print(f"   [Enter]=keep  [r]=redo  [u]=undo last  [s]=skip letter")
            cmd = input("   > ").strip().lower()

            if cmd == 'r':
                # Redo: don't save, don't increment
                print(f"↩️  Redoing sample {i + 1}...")
                continue

            elif cmd == 'u':
                # Undo: remove the most recent accepted sample for this letter
                removed = False
                for idx in range(len(training_data) - 1, -1, -1):
                    if training_data[idx]['letter'] == letter:
                        training_data.pop(idx)
                        i = max(0, i - 1)
                        removed = True
                        print(f"⏪  Undid last '{letter}' — now at [{i + 1}/{samples_per}]")
                        break
                if not removed:
                    print(f"⚠️  Nothing to undo for '{letter}' yet.")
                continue  # redo from the now-vacated slot

            elif cmd == 's':
                # Skip: move on to the next letter without filling all slots
                print(f"⏭️  Skipping remaining samples for '{letter}'")
                break

            else:
                # Keep (Enter or anything unrecognised)
                training_data.append({'letter': letter, 'sensor_data': captured})
                print(f"💾  Saved [{i + 1}/{samples_per}] for '{letter}'")
                i += 1

    # ── Wrap up ────────────────────────────────────────────────────────────
    stop_reader = True
    ser.close()

    # ── Per-letter summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"  Collection Complete!")
    print(f"{'=' * 50}")
    from collections import Counter
    counts = Counter(d['letter'] for d in training_data)
    for ltr in letters:
        print(f"  {ltr} : {counts.get(ltr, 0)} / {samples_per} samples")
    print(f"  Total : {len(training_data)}")

    # ── Revert entire letter(s) before saving ─────────────────────────────
    print(f"\nRevert all samples for a letter before saving?")
    print(f"Type letter (e.g. P) to delete all its samples, or Enter to save:")
    while True:
        cmd = input("  > ").strip().upper()
        if not cmd:
            break
        if cmd in letters:
            before = len(training_data)
            training_data = [d for d in training_data if d['letter'] != cmd]
            removed_count = before - len(training_data)
            print(f"🗑️  Removed all {removed_count} samples for '{cmd}'")
            counts = Counter(d['letter'] for d in training_data)
            for ltr in letters:
                print(f"  {ltr} : {counts.get(ltr, 0)} samples remaining")
            print(f"Type another letter to revert, or Enter to save:")
        else:
            print(f"⚠️  '{cmd}' not in {letters} — try again or press Enter to save.")

    with open('gesture_training_data.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    print("✅ Saved to gesture_training_data.json")

    if training_data:
        print("\n📊 Preview of first sample:")
        print(f"  Letter:      {training_data[0]['letter']}")
        print(f"  Data points: {len(training_data[0]['sensor_data'])}")
        print(f"  First read:  {training_data[0]['sensor_data'][0]}")

    print("\nClose the plot window to exit.")
    plt.show()


if __name__ == '__main__':
    main()