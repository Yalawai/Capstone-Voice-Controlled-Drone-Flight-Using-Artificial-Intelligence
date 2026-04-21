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


def _parse_cm(distance_str: str):
    """Parse a distance string like '150cm' to float. Returns None if unknown."""
    if not distance_str or distance_str == "unknown":
        return None
    try:
        return float(distance_str.lower().replace("cm", "").strip())
    except ValueError:
        return None


def _update_object_distances(object_memory: list, action: str, value_cm: int, current_heading: int):
    """Adjust stored object distances after a movement action."""
    _FORWARD_ACTIONS  = {"move_forward": -1, "move_back": +1}
    _LATERAL_ACTIONS  = {"move_left": -1, "move_right": +1}  # -1 = drone moves left → right-side objects closer
    _VERTICAL_ACTIONS = {"move_up": -1, "move_down": +1}     # -1 = drone moves up → objects above get closer

    for obj in object_memory:
        dist = _parse_cm(obj["distance"])
        if dist is None:
            continue

        rel_angle = (obj["abs_angle"] - current_heading) % 360
        v_angle = obj.get("abs_vertical_angle", 0)

        if action in _FORWARD_ACTIONS:
            if rel_angle <= 45 or rel_angle >= 315:
                dist += _FORWARD_ACTIONS[action] * value_cm
        elif action in _LATERAL_ACTIONS:
            if action == "move_left" and (225 <= rel_angle <= 315):
                dist -= value_cm
            elif action == "move_right" and (45 <= rel_angle <= 135):
                dist -= value_cm
        elif action in _VERTICAL_ACTIONS:
            if action == "move_up":
                if v_angle >= 10:       # object above — getting closer
                    dist -= value_cm
                elif v_angle <= -10:    # object below — getting farther
                    dist += value_cm
            elif action == "move_down":
                if v_angle <= -10:      # object below — getting closer
                    dist -= value_cm
                elif v_angle >= 10:     # object above — getting farther
                    dist += value_cm

        obj["distance"] = f"{max(0, round(dist))}cm"


def _keepalive(stop_event: Event):
    """Sends a no-op command every 5 s so the Tello doesn't auto-land."""
    while not stop_event.wait(5):
        sdk.tello.send_command_without_return("command")


# ── Outer loop: one voice command = one goal run ───────────────────────────────
print("\n===== Drone Control Ready =====")

while not kill_switch.is_set():
    goal = get_voice_command()
    #goal = input("[GOAL]: ")

    if not goal or goal.startswith("("):
        print("[SKIPPED] No valid voice command.")
        continue

    if goal.strip().lower() in ["exit", "                                                                                                   quit"]:
        break

    print(f"\n[GOAL] {goal}")

    sdk.DroneFlightController("takeoff", 0)

    drone_active = True
    history = deque(maxlen=10)
    telemetry = {}
    step = 0
    current_heading = 0        # degrees from mission start, based on executed rotations
    object_memory = []         # persists detected objects across planning cycles
    area_description = ""      # running environment description, updated each cycle

    # ── Inner loop: plan → execute sequence → re-plan until goal done ──────────
    while drone_active and not kill_switch.is_set():

        # 1. Capture image — required for planning
        image_b64 = sdk.TakePicture()
        if not image_b64:
            print("[WARN] Camera failed, retrying in 0.5 s...")
            time.sleep(0.5)
            continue

        telemetry = sdk.DroneSystemInformation() or telemetry

        # 3. Planner: perceive + plan action sequence + check goal
        stop_keepalive = Event()
        keepalive_thread = threading.Thread(target=_keepalive, args=(stop_keepalive,), daemon=True)
        keepalive_thread.start()
        result = vision_planner_agent(goal, image_b64, telemetry, history, object_memory, area_description)
        stop_keepalive.set()
        keepalive_thread.join(timeout=2)

        area_description = result.get("area_description", area_description)

        print(f"\n[PLANNER] {len(result['actions'])} action(s) | "
              f"goal={result['goal_status']} | confidence={result['confidence']:.2f}")
        if area_description:
            print(f"  [AREA] {area_description}")
        if result["perception"]["objects"]:
            for obj in result["perception"]["objects"]:
                abs_ang = (current_heading + obj.get("angle", 0)) % 360
                print(f"  [OBJ] {obj['type']} | {abs_ang}° | {obj['distance']}")
        if result["perception"]["obstacles"]:
            for obs in result["perception"]["obstacles"]:
                abs_ang = (current_heading + obs.get("angle", 0)) % 360
                print(f"  [OBS] obstacle | {abs_ang}° | {obs['distance']}")

        # 4. Goal already achieved or unsafe — stop
        if result["goal_status"] in ("completed", "abort"):
            if result["goal_status"] == "completed":
                print(f"[DONE] Goal completed: {goal}")
            else:
                print(f"[ABORT] {result['goal_reason']}")
            drone_active = False
            break

        # 4b. Update object memory with newly detected objects/obstacles
        seen_this_cycle = set()
        for obj in result["perception"]["objects"] + result["perception"]["obstacles"]:
            obj_type = obj.get("type") or "obstacle"
            abs_angle = (current_heading + obj.get("angle", 0)) % 360
            key = (obj_type, abs_angle)
            if key not in seen_this_cycle:
                seen_this_cycle.add(key)
                # Update existing entry or add new one
                existing = next((o for o in object_memory if o["type"] == obj_type and abs(o["abs_angle"] - abs_angle) <= 15), None)
                if existing:
                    new_dist = obj.get("distance", "unknown")
                    if new_dist != "unknown":
                        existing["distance"] = new_dist
                    existing["abs_vertical_angle"] = obj.get("vertical_angle", 0)
                    existing["step"] = step
                else:
                    object_memory.append({
                        "type": obj_type,
                        "abs_angle": abs_angle,
                        "abs_vertical_angle": obj.get("vertical_angle", 0),
                        "distance": obj.get("distance", "unknown"),
                        "step": step,
                    })

        # 5. Execute each action — avoidance agent checks safety before each move
        for action_item in result["actions"]:

            time.sleep(1)

            # Fresh image for avoidance check
            fresh_image = sdk.TakePicture() or image_b64

            # Avoidance agent — wait for response before moving
            stop_keepalive = Event()
            keepalive_thread = threading.Thread(target=_keepalive, args=(stop_keepalive,), daemon=True)
            keepalive_thread.start()
            avoidance = object_avoidance_agent(action_item, fresh_image, result["perception"])
            stop_keepalive.set()
            keepalive_thread.join(timeout=2)

            if not avoidance["safe"]:
                print(f"[AVOIDANCE] Blocked '{action_item['action']}': {avoidance['reason']} — returning to planner")
                break

            final_action = action_item["action"]
            final_value = int(action_item["value"]) if action_item.get("value") is not None else 0

            print(f"[EXECUTOR] Step {step + 1}: {final_action}" +
                  (f" {final_value}" if final_value else ""))

            sdk.DroneFlightController(final_action, final_value)
            history.append(final_action)
            step += 1

            # Track heading based on executed rotations
            if final_action == "rotate_clockwise":
                current_heading = (current_heading + final_value) % 360
            elif final_action == "rotate_counter_clockwise":
                current_heading = (current_heading - final_value) % 360

            # Update stored object distances based on movement
            _update_object_distances(object_memory, final_action, final_value, current_heading)

        # 6. Replan only if goal not yet met
        if result["goal_status"] == "continue":
            print("[CYCLE] Replanning...\n")

sdk.ShutDown()
print("\n===== Done =====")
