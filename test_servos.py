#!/usr/bin/env python3
"""
test_servos.py
==============

Test script for prosthetic hand servos via PCA9685.
Accepts degrees from -180 to +180.

  0° = center position
  Positive degrees = one direction
  Negative degrees = opposite direction

Usage:
  python3 test_servos.py
"""

import sys
import time

SERVO_AVAILABLE = False
try:
    from adafruit_servokit import ServoKit
    SERVO_AVAILABLE = True
except ImportError:
    print("⚠️  adafruit_servokit not found — running in SIMULATION mode")
    print("   Install with: pip install adafruit-circuitpython-servokit")


# =============================================================================
# CONFIGURATION
# =============================================================================
PCA9685_I2C_ADDRESS = 0x40
SERVO_MIN_PULSE = 500    # µs
SERVO_MAX_PULSE = 2500   # µs
ACTUATION_RANGE = 360    # full range of servo in degrees

SERVOS = [
    {"name": "Thumb",  "channel": 0},
    {"name": "Index",  "channel": 1},
    {"name": "Middle", "channel": 2},
    {"name": "Ring",   "channel": 3},
    {"name": "Little", "channel": 4},
]

CENTER = 180  # physical center = user's 0°
STEP_DELAY = 1.0

# =============================================================================
# SERVO HELPERS
# =============================================================================
kit = None


def init_servos():
    global kit
    if not SERVO_AVAILABLE:
        print("  [SIM] Servos initialized (simulation)")
        return
    try:
        kit = ServoKit(channels=16, address=PCA9685_I2C_ADDRESS)
        for servo in SERVOS:
            ch = servo["channel"]
            kit.servo[ch].actuation_range = ACTUATION_RANGE
            kit.servo[ch].set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)
        print("✅ PCA9685 initialized")
    except Exception as e:
        print(f"❌ Failed to initialize PCA9685: {e}")
        kit = None


def set_servo(channel, degrees, name=""):
    """
    Set servo position in degrees (-180 to +180).
    0° = center, negative = one direction, positive = other.
    """
    degrees = max(-180, min(180, degrees))
    physical = CENTER + degrees  # map to 0-360 range
    physical = max(0, min(360, physical))

    label = f"{name} (ch{channel})" if name else f"ch{channel}"
    print(f"    {label} → {degrees:+4d}° (physical: {physical}°)")

    if kit is not None:
        try:
            kit.servo[channel].angle = physical
        except Exception as e:
            print(f"    ⚠️  Error: {e}")


def set_all(degrees):
    for servo in SERVOS:
        set_servo(servo["channel"], degrees, servo["name"])


# =============================================================================
# TEST 1: Individual
# =============================================================================
def test_individual():
    print("\n" + "=" * 50)
    print("  TEST 1: Individual (0° → +90° → -90° → 0°)")
    print("=" * 50)

    for servo in SERVOS:
        name, ch = servo["name"], servo["channel"]
        print(f"\n  ── {name} (ch{ch}) ──")

        for deg in [0, 90, -90, 0]:
            print(f"  → {deg:+d}°")
            set_servo(ch, deg, name)
            time.sleep(STEP_DELAY)

        print(f"  ✅ {name} OK")

    print("\n✅ Individual test complete!")


# =============================================================================
# TEST 2: All together
# =============================================================================
def test_all_together():
    print("\n" + "=" * 50)
    print("  TEST 2: All Together (0° → +90° → -90° → 0°)")
    print("=" * 50)

    for deg in [0, 90, -90, 0]:
        print(f"\n  → All to {deg:+d}°")
        set_all(deg)
        time.sleep(STEP_DELAY * 1.5)

    print("\n✅ All-together test complete!")


# =============================================================================
# TEST 3: Interactive
# =============================================================================
def test_interactive():
    print("\n" + "=" * 50)
    print("  TEST 3: Interactive Mode")
    print("=" * 50)
    print("""
  Commands:
    <servo#> <degrees>  — e.g. '0 90' or '1 -45'
    all <degrees>       — e.g. 'all -90'
    rest                — All to 0°
    q                   — Quit

  Degrees: -180 to +180 (0 = center)

  Servos:""")
    for s in SERVOS:
        print(f"    {s['channel']} = {s['name']}")
    print()

    while True:
        try:
            cmd = input("  > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if not cmd:
            continue
        if cmd in ("q", "quit", "exit"):
            break
        if cmd == "rest":
            set_all(0)
            continue

        parts = cmd.split()
        if len(parts) != 2:
            print("  ⚠️  Usage: <servo#> <degrees>  or  all <degrees>")
            continue

        try:
            deg = int(parts[1])
        except ValueError:
            print("  ⚠️  Degrees must be a number (-180 to 180)")
            continue

        if deg < -180 or deg > 180:
            print("  ⚠️  Range: -180 to 180")
            continue

        if parts[0] == "all":
            set_all(deg)
        else:
            try:
                ch = int(parts[0])
            except ValueError:
                print("  ⚠️  Servo must be 0-4")
                continue
            if ch < 0 or ch > 4:
                print("  ⚠️  Servo must be 0-4")
                continue
            set_servo(ch, deg, SERVOS[ch]["name"])

    set_all(0)
    print("  ✅ Done")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n🦾 SERVO TEST")
    print("=" * 50)

    init_servos()
    print("\n🔄 All to center (0°)...")
    set_all(0)
    time.sleep(0.5)

    while True:
        print("\n" + "─" * 50)
        print("  [1] Individual test (one by one)")
        print("  [2] All together")
        print("  [3] Interactive (type degrees)")
        print("  [0] Exit")
        print("─" * 50)

        try:
            choice = input("  Choice: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if choice == "1": test_individual()
        elif choice == "2": test_all_together()
        elif choice == "3": test_interactive()
        elif choice == "0": break

    print("\n🔄 All to center...")
    set_all(0)
    print("👋 Done!")


if __name__ == "__main__":
    main()
