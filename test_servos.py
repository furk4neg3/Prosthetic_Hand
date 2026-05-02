#!/usr/bin/env python3
"""
test_servos.py
==============

Test script for MG90S continuous rotation servos via PCA9685.

Continuous rotation servos work differently from positional servos:
  - They DON'T go to a specific angle
  - Instead, the pulse width controls SPEED and DIRECTION
  - ~1500µs = STOP (dead zone)
  - < 1500µs = rotate one direction (CW), speed increases as value decreases
  - > 1500µs = rotate other direction (CCW), speed increases as value increases

This script lets you control servos using a throttle value:
  -100 = full speed direction A
     0 = stop
  +100 = full speed direction B

Usage:
  python3 test_servos.py
"""

import sys
import time

# ─────────────────────────────────────────────────────────────────────────────
# Try to import servo control libraries
# ─────────────────────────────────────────────────────────────────────────────
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
PCA9685_FREQUENCY = 50

SERVOS = [
    {"name": "Thumb",  "channel": 0},
    {"name": "Index",  "channel": 1},
    {"name": "Middle", "channel": 2},
    {"name": "Ring",   "channel": 3},
    {"name": "Little", "channel": 4},
]

# How long to run each servo during auto tests (seconds)
RUN_DURATION = 1.0


# =============================================================================
# SERVO HELPERS
# =============================================================================
kit = None


def init_servos():
    """Initialize the PCA9685."""
    global kit
    if not SERVO_AVAILABLE:
        print("  [SIM] Servos initialized (simulation)")
        return

    try:
        kit = ServoKit(
            channels=16,
            address=PCA9685_I2C_ADDRESS,
        )
        print("✅ PCA9685 initialized")
    except Exception as e:
        print(f"❌ Failed to initialize PCA9685: {e}")
        kit = None


def set_throttle(channel, throttle, name=""):
    """
    Set a continuous rotation servo's throttle.

    throttle: float from -1.0 to 1.0
      -1.0 = full speed direction A
       0.0 = stop
      +1.0 = full speed direction B
    """
    throttle = max(-1.0, min(1.0, throttle))
    label = f"{name} (ch{channel})" if name else f"ch{channel}"

    if throttle == 0:
        print(f"    {label} → STOP")
    elif throttle > 0:
        print(f"    {label} → +{throttle:.2f} (direction A, {abs(throttle)*100:.0f}% speed)")
    else:
        print(f"    {label} → {throttle:.2f} (direction B, {abs(throttle)*100:.0f}% speed)")

    if kit is not None:
        try:
            kit.continuous_servo[channel].throttle = throttle
        except Exception as e:
            print(f"    ⚠️  Error: {e}")


def stop_all():
    """Stop all servos."""
    for servo in SERVOS:
        set_throttle(servo["channel"], 0, servo["name"])


# =============================================================================
# TEST 1: Individual servo test
# =============================================================================
def test_individual():
    """Test each servo one by one in both directions."""
    print()
    print("=" * 50)
    print("  TEST 1: Individual Servo Test")
    print(f"  Each servo: stop → CW → stop → CCW → stop")
    print(f"  Duration per step: {RUN_DURATION}s")
    print("=" * 50)

    for servo in SERVOS:
        name = servo["name"]
        ch = servo["channel"]

        print(f"\n  ── {name} (channel {ch}) ──")

        print(f"  → Stop")
        set_throttle(ch, 0, name)
        time.sleep(RUN_DURATION)

        print(f"  → Direction A (half speed)")
        set_throttle(ch, 0.5, name)
        time.sleep(RUN_DURATION)

        print(f"  → Stop")
        set_throttle(ch, 0, name)
        time.sleep(RUN_DURATION)

        print(f"  → Direction B (half speed)")
        set_throttle(ch, -0.5, name)
        time.sleep(RUN_DURATION)

        print(f"  → Stop")
        set_throttle(ch, 0, name)
        time.sleep(0.5)

        print(f"  ✅ {name} done")

    print("\n✅ Individual test complete!")


# =============================================================================
# TEST 2: All servos together
# =============================================================================
def test_all_together():
    """Test all servos moving together."""
    print()
    print("=" * 50)
    print("  TEST 2: All Servos Together")
    print(f"  Duration per step: {RUN_DURATION}s")
    print("=" * 50)

    print(f"\n  → All STOP")
    stop_all()
    time.sleep(RUN_DURATION)

    print(f"\n  → All direction A (half speed)")
    for servo in SERVOS:
        set_throttle(servo["channel"], 0.5, servo["name"])
    time.sleep(RUN_DURATION)

    print(f"\n  → All STOP")
    stop_all()
    time.sleep(RUN_DURATION)

    print(f"\n  → All direction B (half speed)")
    for servo in SERVOS:
        set_throttle(servo["channel"], -0.5, servo["name"])
    time.sleep(RUN_DURATION)

    print(f"\n  → All STOP")
    stop_all()

    print("\n✅ All-together test complete!")


# =============================================================================
# TEST 3: Interactive mode
# =============================================================================
def test_interactive():
    """Manual control of any servo."""
    print()
    print("=" * 50)
    print("  TEST 3: Interactive Mode")
    print("=" * 50)

    print("""
  Commands:
    <servo#> <throttle>  — Set servo throttle (-100 to +100)
                           e.g. '0 50'  = servo 0 at 50% direction A
                           e.g. '1 -75' = servo 1 at 75% direction B
    all <throttle>       — Set ALL servos
    stop                 — Stop all servos
    q                    — Quit

  Servo numbers:""")
    for servo in SERVOS:
        print(f"    {servo['channel']} = {servo['name']}")
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

        if cmd == "stop":
            print("  → Stopping all")
            stop_all()
            continue

        parts = cmd.split()
        if len(parts) != 2:
            print("  ⚠️  Usage: <servo#> <throttle>  or  all <throttle>  or  stop  or  q")
            continue

        try:
            throttle_pct = float(parts[1])
        except ValueError:
            print("  ⚠️  Throttle must be a number (-100 to +100)")
            continue

        if throttle_pct < -100 or throttle_pct > 100:
            print("  ⚠️  Throttle must be between -100 and +100")
            continue

        throttle = throttle_pct / 100.0  # Convert to -1.0 to 1.0

        if parts[0] == "all":
            print(f"  → All servos to {throttle_pct:+.0f}%")
            for servo in SERVOS:
                set_throttle(servo["channel"], throttle, servo["name"])
        else:
            try:
                ch = int(parts[0])
            except ValueError:
                print("  ⚠️  Servo must be a number (0-4)")
                continue

            if ch < 0 or ch > 4:
                print("  ⚠️  Servo must be between 0 and 4")
                continue

            name = SERVOS[ch]["name"]
            set_throttle(ch, throttle, name)

    stop_all()
    print("  ✅ Interactive mode ended")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print()
    print("🦾 PROSTHETIC HAND — SERVO TEST (Continuous Rotation)")
    print("=" * 50)

    init_servos()

    print("\n🔄 Stopping all servos...")
    stop_all()
    time.sleep(0.5)

    while True:
        print()
        print("─" * 50)
        print("  Select a test:")
        print("  [1] Individual servo test (one by one, both directions)")
        print("  [2] All servos together")
        print("  [3] Interactive mode (manual throttle control)")
        print("  [0] Exit")
        print("─" * 50)

        try:
            choice = input("  Choice: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if choice == "1":
            test_individual()
        elif choice == "2":
            test_all_together()
        elif choice == "3":
            test_interactive()
        elif choice == "0":
            break
        else:
            print("  ⚠️  Enter 0, 1, 2, or 3")

    print("\n🔄 Stopping all servos...")
    stop_all()
    print("👋 Done!")


if __name__ == "__main__":
    main()
