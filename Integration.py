import copy
import threading
import time
from collections import deque
from threading import Event

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

    # ── Per-mission state ──────────────────────────────────────────────────
    mission = {
        "history":         deque(maxlen=10),
        "telemetry":       {},
        "object_memory":   [],
        "area_description": "",
        "current_heading": 0,
        "step":            0,
    }

    # ── Keepalive thread (planner is now synchronous, no planner thread) ───
    stop_keepalive = Event()
    keepalive_thread = threading.Thread(target=_keepalive, args=(stop_keepalive,), daemon=True)
    keepalive_thread.start()

    plan_seq = 0

    def _call_planner(current_plan_actions, action_idx):
        """Take a picture and call the planner. Returns the result dict or None on failure."""
        image = sdk.TakePicture()
        if not image:
            return None
        try:
            return vision_planner_agent(
                goal, image, dict(mission["telemetry"]),
                list(mission["history"]), list(mission["object_memory"]),
                mission["area_description"],
                current_plan_actions=copy.deepcopy(current_plan_actions),
                action_idx=action_idx,
            )
        except Exception as e:
            print("[PLANNER] Error:", e)
            return None

    def _apply_plan_state(plan, seq_num):
        """Refresh telemetry, area, object memory, and print summary for a plan.
        Returns (risk_level, goal_status)."""
        perception = plan["perception"]
        risk_level = perception.get("risk_level", "low")
        goal_status = plan["goal_status"]

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

    # ── Initial plan ────────────────────────────────────────────────────────
    print("[EXECUTOR] Requesting initial plan...")
    current_plan = None
    while current_plan is None and not kill_switch.is_set():
        current_plan = _call_planner(None, 0)
        if current_plan is None:
            time.sleep(0.5)

    drone_active = not kill_switch.is_set()
    action_idx = 0
    risk_level = "low"

    if drone_active:
        plan_seq += 1
        risk_level, goal_status = _apply_plan_state(current_plan, plan_seq)
        if goal_status in ("completed", "abort"):
            if goal_status == "completed":
                print(f"[DONE] Goal completed: {goal}")
            else:
                print(f"[ABORT] {current_plan['goal_reason']}")
            drone_active = False

    # ── Executor loop: API only at end-of-plan or api_check actions ────────
    while drone_active and not kill_switch.is_set():

        # Out of actions in current plan — call planner for a new plan
        if action_idx >= len(current_plan["actions"]):
            print("[EXECUTOR] Plan exhausted — requesting new plan...")
            new_plan = _call_planner(None, 0)
            if new_plan is None:
                time.sleep(0.5)
                continue
            plan_seq += 1
            risk_level, goal_status = _apply_plan_state(new_plan, plan_seq)
            if goal_status in ("completed", "abort"):
                if goal_status == "completed":
                    print(f"[DONE] Goal completed: {goal}")
                else:
                    print(f"[ABORT] {new_plan['goal_reason']}")
                drone_active = False
                break
            current_plan = new_plan
            action_idx = 0
            print(f"[EXECUTOR] New plan ({len(current_plan['actions'])} actions)")
            continue

        action_item = current_plan["actions"][action_idx]

        # api_check — call planner now to refresh perception mid-plan
        if action_item["action"] == "api_check":
            print(f"[EXECUTOR] api_check at step {action_idx + 1} — calling planner...")
            new_plan = _call_planner(current_plan["actions"], action_idx + 1)
            if new_plan is None:
                print("[EXECUTOR] Planner failed — hovering and skipping api_check")
                sdk.DroneFlightController("hover", 0)
                action_idx += 1
                continue
            plan_seq += 1
            risk_level, goal_status = _apply_plan_state(new_plan, plan_seq)
            if goal_status in ("completed", "abort"):
                if goal_status == "completed":
                    print(f"[DONE] Goal completed: {goal}")
                else:
                    print(f"[ABORT] {new_plan['goal_reason']}")
                drone_active = False
                break
            decision = new_plan.get("plan_decision", "replace")
            if decision == "replace":
                current_plan = new_plan
                action_idx = 0
                print(f"[EXECUTOR] Plan replaced ({len(current_plan['actions'])} actions)")
            else:
                remaining = len(current_plan["actions"]) - (action_idx + 1)
                print(f"[EXECUTOR] Plan kept; {remaining} action(s) remaining after api_check")
                action_idx += 1
            continue

        # Execute the action
        time.sleep(1)
        final_action = action_item["action"]
        final_value  = int(action_item["value"]) if action_item.get("value") is not None else 0

        # Cap rotation to 5° when lining up with an object
        _ALIGN_KEYWORDS = {"align", "crosshair", "line up", "lineup", "center"}
        reason_lower = action_item.get("reason", "").lower()
        if final_action in ("rotate_clockwise", "rotate_counter_clockwise") and \
                any(kw in reason_lower for kw in _ALIGN_KEYWORDS):
            final_value = min(final_value, 5)

        step = mission["step"]
        print(f"[EXECUTOR] Step {step + 1}: {final_action}" +
              (f" {final_value}" if final_value else "") +
              f" — {action_item.get('reason', '')}")

        sdk.DroneFlightController(final_action, final_value)

        action_idx += 1

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

    # ── Mission done — stop keepalive ──────────────────────────────────────
    stop_keepalive.set()
    keepalive_thread.join(timeout=2)

sdk.ShutDown()
print("\n===== Done =====")
