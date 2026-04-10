import threading
import msvcrt
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from tello_sdk_controls_dir.main import SDK
from whisper_cpp.main import main as get_voice_command
from vision_action_controller_dir.main import vision_planner_agent

# ── SDK ────────────────────────────────────────────────────────────────────────
sdk = SDK()

# ── Kill switch ────────────────────────────────────────────────────────────────
def kill_listener():
    print("Press '!' to EMERGENCY KILL")
    while True:
        if msvcrt.kbhit():
            if msvcrt.getwch() == '!':
                print("[SAFETY] KILL SWITCH TRIGGERED")
                sdk.emergency_kill()
                break
        time.sleep(0.05)
threading.Thread(target=kill_listener, daemon=True).start()

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
    history = []
    perception = {}
    telemetry = {}
    step = 0

    # ── Inner loop: control cycle ──────────────────────────────────────────────
    while drone_active:
        # 1. Fetch image + telemetry in parallel
        with ThreadPoolExecutor(max_workers=2) as ex:
            pic_future = ex.submit(sdk.TakePicture)
            tel_future = ex.submit(sdk.DroneSystemInformation)
            image_b64 = pic_future.result()
            telemetry = tel_future.result()

        # 2. Vision + planning + goal check (single LLM call)
        # Keep the Tello alive during the LLM call (drone auto-lands after 15s without commands)
        stop_keepalive = Event()
        keepalive_thread = threading.Thread(target=sdk.send_keepalive, args=(stop_keepalive,), daemon=True)
        keepalive_thread.start()
        result = vision_planner_agent(goal, image_b64, telemetry, history)
        stop_keepalive.set()
        perception = result["perception"]
        action = result["action"]
        check = result["goal_check"]

        # 3. Execute action via SDK
        action_name = action["action"]
        value = int(action["value"]) if action.get("value") is not None else 0
        print(f"\n[EXECUTOR] ───── Step {step + 1} ─────")
        print(f"[EXECUTOR] Action: {action_name}" + (f" | Value: {value}" if value else ""))
        print(f"[EXECUTOR] Reason: {action['reason']} | Confidence: {action['confidence']}")
        sdk.DroneFlightController(action_name, value)
        history.append(action_name)
        step += 1

        # 4. Check goal (now evaluated every step via combined LLM call)
        print(f"[GOAL CHECK] Status: {check.status} | Reason: {check.reason}")
        if check.status in ["completed", "abort"]:
            if check.status == "completed":
                print(f"[DONE] Goal completed: {goal}")
            else:
                print(f"[ABORT] {check.reason}")
            drone_active = False
sdk.ShutDown()
print("\n===== Done =====")