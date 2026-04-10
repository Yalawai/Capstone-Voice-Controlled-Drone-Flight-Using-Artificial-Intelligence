import threading
import time
from collections import deque
from threading import Event

from tello_sdk_controls_dir.main import SDK
from whisper_cpp.main import main as get_voice_command
from vision_action_controller_dir.main import vision_planner_agent

# ── SDK ────────────────────────────────────────────────────────────────────────
sdk = SDK()

def _keepalive(stop_event):
    while not stop_event.wait(10):
        sdk.tello.send_command_without_return("command")

print("\n===== Drone Control Ready =====")

# ── Outer loop: one voice command = one goal run ───────────────────────────────
while True:
    #goal = get_voice_command()
    goal = input("[GOAL]: ")

    # Skip empty or failed transcriptions
    if not goal or goal.startswith("("):
        print("[SKIPPED] No valid voice command.")
        continue

    if goal.strip().lower() in ["exit", "quit"]:
        break

    print(f"\n[GOAL] {goal}")

    drone_active = True
    history = deque(maxlen=10)
    perception = {}
    telemetry = {}
    step = 0
    last_call_time = 0.0
    MIN_CALL_INTERVAL = 1.0  # seconds between LLM calls

    # ── Inner loop: control cycle ──────────────────────────────────────────────
    while drone_active:
        # 1. Fetch image + telemetry every iteration
        image_b64 = sdk.TakePicture()
        telemetry = sdk.DroneSystemInformation() or telemetry

        # 2. Rate-limit LLM calls
        elapsed = time.time() - last_call_time
        if elapsed < MIN_CALL_INTERVAL:
            time.sleep(MIN_CALL_INTERVAL - elapsed)

        # 3. Vision + planning + goal check (single LLM call)
        # Keep the Tello alive during the LLM call (drone auto-lands after 15s without commands)
        stop_keepalive = Event()
        keepalive_thread = threading.Thread(target=_keepalive, args=(stop_keepalive,), daemon=True)
        keepalive_thread.start()
        last_call_time = time.time()
        result = vision_planner_agent(goal, image_b64, telemetry, history)
        stop_keepalive.set()
        keepalive_thread.join(timeout=2)
        perception = result["perception"]
        action = result["action"]
        check = result["goal_check"]

        # 4. Execute action via SDK
        action_name = action["action"]
        value = int(action["value"]) if action.get("value") is not None else 0
        print(f"\n[EXECUTOR] ───── Step {step + 1} ─────")
        print(f"[EXECUTOR] Action: {action_name}" + (f" | Value: {value}" if value else ""))
        print(f"[EXECUTOR] Reason: {action['reason']} | Confidence: {action['confidence']}")
        sdk.DroneFlightController(action_name, value)
        history.append(action_name)
        step += 1

        # 5. Check goal (now evaluated every step via combined LLM call)
        print(f"[GOAL CHECK] Status: {check.status} | Reason: {check.reason}")
        if check.status in ["completed", "abort"]:
            if check.status == "completed":
                print(f"[DONE] Goal completed: {goal}")
            else:
                print(f"[ABORT] {check.reason}")
            drone_active = False
sdk.ShutDown()
print("\n===== Done =====")