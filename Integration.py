import copy
import threading
import time
from collections import deque
from threading import Event, Lock

import keyboard

from tello_sdk_controls_dir.main import SDK
from whisper_cpp.main import main as get_voice_command
from vision_action_controller_dir.main import vision_planner_agent

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
    _LATERAL_ACTIONS  = {"move_left": -1, "move_right": +1}
    _VERTICAL_ACTIONS = {"move_up": -1, "move_down": +1}

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
                if v_angle >= 10:
                    dist -= value_cm
                elif v_angle <= -10:
                    dist += value_cm
            elif action == "move_down":
                if v_angle <= -10:
                    dist -= value_cm
                elif v_angle >= 10:
                    dist += value_cm

        obj["distance"] = f"{max(0, round(dist))}cm"


def _keepalive(stop_event: Event):
    """Sends a no-op command every 5 s so the Tello doesn't auto-land."""
    while not stop_event.wait(5):
        sdk.tello.send_command_without_return("command")


# ── Outer loop: one voice command = one goal run ───────────────────────────────
print("\n===== Drone Control Ready =====")

while not kill_switch.is_set():
    #goal = get_voice_command()
    goal = input("[GOAL]: ")

    if not goal or goal.startswith("("):
        print("[SKIPPED] No valid voice command.")
        continue

    if goal.strip().lower() in ["exit", "quit"]:
        break

    print(f"\n[GOAL] {goal}")
    sdk.DroneFlightController("takeoff", 0)
    # ── Per-mission shared state (executor thread owns writes) ─────────────
    state_lock = Lock()
    mission = {
        "history":         deque(maxlen=10),
        "telemetry":       {},
        "object_memory":   [],
        "area_description": "",
        "current_heading": 0,
        "step":            0,
        "current_plan_actions": None,  # list of action dicts the executor is working through
        "action_idx":      0,          # index of the next action to be executed
    }

    # ── Planner output (planner thread writes, executor reads) ─────────────
    plan_lock  = Lock()
    plan_box   = [None]   # plan_box[0] = latest vision_planner_agent result
    plan_seq   = [0]      # incremented on every new plan
    plan_event = Event()  # fired each time a fresh plan arrives
    stop_planner = Event()

    def _planner_worker():
        """Background thread: calls vision_planner_agent in a tight loop."""
        while not stop_planner.is_set() and not kill_switch.is_set():
            # Snapshot mission state for this call
            with state_lock:
                history_snap  = list(mission["history"])
                telemetry_snap = dict(mission["telemetry"])
                obj_mem_snap  = list(mission["object_memory"])
                area_desc_snap = mission["area_description"]
                plan_actions_snap = (
                    copy.deepcopy(mission["current_plan_actions"])
                    if mission["current_plan_actions"] is not None else None
                )
                action_idx_snap = mission["action_idx"]

            image = sdk.TakePicture()
            if not image:
                time.sleep(0.3)
                continue

            try:
                result = vision_planner_agent(
                    goal, image, telemetry_snap,
                    history_snap, obj_mem_snap, area_desc_snap,
                    current_plan_actions=plan_actions_snap,
                    action_idx=action_idx_snap,
                )
            except Exception as e:
                print("[PLANNER THREAD] Error:", e)
                time.sleep(1)
                continue

            with plan_lock:
                plan_box[0] = result
                plan_seq[0] += 1
            plan_event.set()  # wake up anything waiting for a fresh plan

    # ── Start background threads ────────────────────────────────────────────
    planner_thread = threading.Thread(target=_planner_worker, daemon=True)
    planner_thread.start()

    stop_keepalive = Event()
    keepalive_thread = threading.Thread(target=_keepalive, args=(stop_keepalive,), daemon=True)
    keepalive_thread.start()

    def _apply_plan_state(plan, seq_num):
        """Refresh telemetry, area, object memory, and print summary for a plan.
        Returns (risk_level, goal_status)."""
        perception = plan["perception"]
        risk_level = perception.get("risk_level", "low")
        goal_status = plan["goal_status"]

        with state_lock:
            new_tel = sdk.DroneSystemInformation()
            if new_tel:
                mission["telemetry"] = new_tel
            mission["area_description"] = plan.get("area_description", mission["area_description"])
            step = mission["step"]
            ch   = mission["current_heading"]
            ad   = mission["area_description"]
            obj_mem = mission["object_memory"]

        print(f"\n[PLAN #{seq_num}] {len(plan['actions'])} action(s) | "
              f"goal={goal_status} | risk={risk_level} | "
              f"confidence={plan['confidence']:.2f}")
        if ad:
            print(f"  [AREA] {ad}")
        for obj in perception.get("objects", []):
            print(f"  [OBJ] {obj['type']} | {(ch + obj.get('angle', 0)) % 360}° | {obj['distance']}")
        for obs in perception.get("obstacles", []):
            print(f"  [OBS] obstacle | {(ch + obs.get('angle', 0)) % 360}° | {obs['distance']}")
        if plan.get("message_to_user"):
            print(f"  [MSG] {plan['message_to_user']}")

        seen_this_cycle = set()
        for obj in perception.get("objects", []) + perception.get("obstacles", []):
            obj_type  = obj.get("type") or "obstacle"
            abs_angle = (ch + obj.get("angle", 0)) % 360
            key = (obj_type, abs_angle)
            if key not in seen_this_cycle:
                seen_this_cycle.add(key)
                existing = next(
                    (o for o in obj_mem if o["type"] == obj_type and abs(o["abs_angle"] - abs_angle) <= 15),
                    None,
                )
                if existing:
                    new_dist = obj.get("distance", "unknown")
                    if new_dist != "unknown":
                        existing["distance"] = new_dist
                    existing["abs_vertical_angle"] = obj.get("vertical_angle", 0)
                    existing["step"] = step
                else:
                    obj_mem.append({
                        "type":              obj_type,
                        "abs_angle":         abs_angle,
                        "abs_vertical_angle": obj.get("vertical_angle", 0),
                        "distance":          obj.get("distance", "unknown"),
                        "step":              step,
                    })

        return risk_level, goal_status

    # Wait for the very first plan before beginning execution
    print("[EXECUTOR] Waiting for first plan...")
    plan_event.wait(timeout=30)

    drone_active = True
    current_plan = None
    last_exec_seq = -1
    action_idx = 0
    risk_level = "low"

    # ── Executor loop: planner emits keep/replace decisions; executor honours them ─
    while drone_active and not kill_switch.is_set():

        # 1. Pick up any new planner result and apply its keep/replace decision
        with plan_lock:
            live_seq = plan_seq[0]
            new_result_available = (plan_box[0] is not None and live_seq != last_exec_seq)
            if new_result_available:
                new_result = copy.deepcopy(plan_box[0])
                last_exec_seq = live_seq
            else:
                new_result = None

        if new_result_available:
            decision = new_result.get("plan_decision", "replace")
            risk_level, goal_status = _apply_plan_state(new_result, last_exec_seq)

            if goal_status in ("completed", "abort"):
                if goal_status == "completed":
                    print(f"[DONE] Goal completed: {goal}")
                else:
                    print(f"[ABORT] {new_result['goal_reason']}")
                drone_active = False
                break

            if decision == "replace" or current_plan is None:
                current_plan = new_result
                action_idx = 0
                print(f"[EXECUTOR] Plan replaced ({len(current_plan['actions'])} actions)")
            else:
                remaining = len(current_plan["actions"]) - action_idx
                print(f"[EXECUTOR] Plan kept; {remaining} action(s) remaining")

            with state_lock:
                mission["current_plan_actions"] = list(current_plan["actions"])
                mission["action_idx"] = action_idx

        if current_plan is None:
            print("[EXECUTOR] No plan yet — waiting...")
            plan_event.clear()
            plan_event.wait(timeout=15)
            continue

        # 2. Out of actions in current plan — wait for the next planner result
        if action_idx >= len(current_plan["actions"]):
            plan_event.clear()
            plan_event.wait(timeout=15)
            continue

        action_item = current_plan["actions"][action_idx]

        # 3. api_check — block for one fresh planner cycle, then advance past it.
        #    The next loop iteration will pick up the new plan; if it's a "replace"
        #    that resets action_idx, otherwise we just continue with the next action.
        if action_item["action"] == "api_check":
            print(f"[EXECUTOR] api_check at step {action_idx + 1} — waiting for fresh plan...")
            with plan_lock:
                seq_before = plan_seq[0]
            plan_event.clear()
            got = plan_event.wait(timeout=20)
            with plan_lock:
                fresh_seq = plan_seq[0]
            if not got or fresh_seq == seq_before:
                print("[EXECUTOR] Timeout waiting for fresh plan — hovering")
                sdk.DroneFlightController("hover", 0)
            action_idx += 1
            with state_lock:
                mission["action_idx"] = action_idx
            continue

        # 4. Execute the action
        time.sleep(1)
        final_action = action_item["action"]
        final_value  = int(action_item["value"]) if action_item.get("value") is not None else 0

        # Cap rotation to 5° when lining up with an object
        _ALIGN_KEYWORDS = {"align", "crosshair", "line up", "lineup", "center"}
        reason_lower = action_item.get("reason", "").lower()
        if final_action in ("rotate_clockwise", "rotate_counter_clockwise") and \
                any(kw in reason_lower for kw in _ALIGN_KEYWORDS):
            final_value = min(final_value, 5)

        with state_lock:
            step = mission["step"]
        print(f"[EXECUTOR] Step {step + 1}: {final_action}" +
              (f" {final_value}" if final_value else "") +
              f" — {action_item.get('reason', '')}")

        sdk.DroneFlightController(final_action, final_value)

        action_idx += 1

        with state_lock:
            mission["history"].append({
                "action": final_action,
                "value": final_value,
                "reason": action_item.get("reason", ""),
            })
            mission["step"] += 1
            if final_action == "rotate_clockwise":
                mission["current_heading"] = (mission["current_heading"] + final_value) % 360
            elif final_action == "rotate_counter_clockwise":
                mission["current_heading"] = (mission["current_heading"] - final_value) % 360
            ch = mission["current_heading"]
            _update_object_distances(mission["object_memory"], final_action, final_value, ch)
            mission["action_idx"] = action_idx

    # ── Mission done — stop background threads ──────────────────────────────
    stop_planner.set()
    stop_keepalive.set()
    planner_thread.join(timeout=5)
    keepalive_thread.join(timeout=2)

sdk.ShutDown()
print("\n===== Done =====")
