import threading
import time
from collections import deque
from threading import Event

import keyboard

from tello_sdk_controls_dir.main import SDK
from whisper_cpp.main import main as get_voice_command
from vision_action_controller_dir.main import vision_planner_agent, object_avoidance_agent

# ── SDK ────────────────────────────────────────────────────────────────────────
sdk = SDK()

MIN_CALL_INTERVAL = 1  # minimum seconds between LLM calls

# ── Kill switch ────────────────────────────────────────────────────────────────
kill_switch = Event()

def _on_kill_switch():
    print("\n[KILL SWITCH] ESC pressed — landing immediately!")
    kill_switch.set()
    try:
        sdk.tello.land()
    except Exception as e:
        print("[KILL SWITCH] Land failed:", e)

keyboard.add_hotkey("esc", _on_kill_switch)
print("Kill switch active — press ESC at any time to land immediately.")


def _keepalive(stop_event: Event):
    """Sends a no-op command every 10 s so the Tello doesn't auto-land."""
    while not stop_event.wait(10):
        sdk.tello.send_command_without_return("command")


# ── Outer loop: one voice command = one goal run ───────────────────────────────
print("\n===== Drone Control Ready =====")

while not kill_switch.is_set():
    # goal = get_voice_command()
    goal = input("[GOAL]: ")

    if not goal or goal.startswith("("):
        print("[SKIPPED] No valid voice command.")
        continue

    if goal.strip().lower() in ["exit", "quit"]:
        break

    print(f"\n[GOAL] {goal}")

    sdk.DroneFlightController("takeoff", 0)

    drone_active = True
    history = deque(maxlen=10)
    telemetry = {}
    step = 0
    last_call_time = 0.0

    # ── Inner loop: plan → execute sequence → re-plan until goal done ──────────
    while drone_active and not kill_switch.is_set():

        # 1. Capture image — required for planning
        image_b64 = sdk.TakePicture()
        if not image_b64:
            print("[WARN] Camera failed, retrying in 0.5 s...")
            time.sleep(0.5)
            continue

        telemetry = sdk.DroneSystemInformation() or telemetry

        # 2. Rate-limit LLM calls
        elapsed = time.time() - last_call_time
        if elapsed < MIN_CALL_INTERVAL:
            time.sleep(MIN_CALL_INTERVAL - elapsed)

        # 3. Planner: perceive + plan action sequence + check goal
        stop_keepalive = Event()
        keepalive_thread = threading.Thread(target=_keepalive, args=(stop_keepalive,), daemon=True)
        keepalive_thread.start()
        last_call_time = time.time()
        result = vision_planner_agent(goal, image_b64, telemetry, history)
        stop_keepalive.set()
        keepalive_thread.join(timeout=2)

        print(f"\n[PLANNER] {len(result['actions'])} action(s) | "
              f"goal={result['goal_status']} | confidence={result['confidence']:.2f}")

        # 4. Goal already achieved or unsafe — stop
        if result["goal_status"] in ("completed", "abort"):
            if result["goal_status"] == "completed":
                print(f"[DONE] Goal completed: {goal}")
            else:
                print(f"[ABORT] {result['goal_reason']}")
            drone_active = False
            break

        # 5. Execute each action — avoidance agent checks safety before each move
        for action_item in result["actions"]:

            # Fresh image for avoidance check
            fresh_image = sdk.TakePicture() or image_b64

            # Avoidance agent — wait for response before moving
            stop_keepalive = Event()
            keepalive_thread = threading.Thread(target=_keepalive, args=(stop_keepalive,), daemon=True)
            keepalive_thread.start()
            avoidance = object_avoidance_agent(action_item, fresh_image)
            stop_keepalive.set()
            keepalive_thread.join(timeout=2)

            if not avoidance["safe"]:
                print(f"[AVOIDANCE] Skipping '{action_item['action']}': {avoidance['reason']}")
                continue

            final_action = action_item["action"]
            final_value = int(action_item["value"]) if action_item.get("value") is not None else 0

            print(f"[EXECUTOR] Step {step + 1}: {final_action}" +
                  (f" {final_value}" if final_value else ""))

            sdk.DroneFlightController(final_action, final_value)
            history.append(final_action)
            step += 1

        # 6. Sequence done — re-plan to check goal progress
        print("[CYCLE] Sequence complete — re-planning...\n")

sdk.ShutDown()
print("\n===== Done =====")
