from typing import List, Optional, Literal
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from pydantic import BaseModel

load_dotenv()


# ── Shared sub-models ──────────────────────────────────────────────────────────

class ObjectItem(BaseModel):
    type: str
    angle: int           # horizontal relative angle from drone forward, negative=left, positive=right, range -41 to +41
    vertical_angle: int  # vertical relative angle, negative=below center, positive=above center, range -30 to +30
    distance: str        # estimated distance in cm e.g. "150cm", or "unknown"

class ObstacleItem(BaseModel):
    angle: int           # horizontal relative angle from drone forward, negative=left, positive=right, range -41 to +41
    vertical_angle: int  # vertical relative angle, negative=below center, positive=above center, range -30 to +30
    distance: str        # estimated distance in cm e.g. "150cm", or "unknown"

_ACTION_LITERAL = Literal[
    "takeoff", "land", "hover",
    "move_forward", "move_back", "move_left", "move_right",
    "move_up", "move_down",
    "rotate_clockwise", "rotate_counter_clockwise",
    "api_check"
]

class ActionItem(BaseModel):
    action: _ACTION_LITERAL
    value: Optional[float] = None   # cm for movement, degrees for rotation, null otherwise
    reason: str


# ── Planner output ─────────────────────────────────────────────────────────────

class PlannerOutput(BaseModel):
    objects: List[ObjectItem]
    obstacles: List[ObstacleItem]
    free_space: List[Literal["left", "center", "right"]]
    environment: Literal["indoor", "outdoor", "unknown"]
    risk_level: Literal["low", "medium", "high"]
    plan_decision: Literal["keep", "replace"]   # keep current in-flight plan, or replace it with `actions`
    actions: List[ActionItem]       # ordered sequence of 3-10 actions; ignored when plan_decision == "keep"
    confidence: float
    goal_status: Literal["continue", "completed", "abort"]
    goal_reason: str
    area_description: str           # running description of the environment, updated each cycle
    message_to_user: Optional[str] = None


# ── LLMs ──────────────────────────────────────────────────────────────────────

_planner_llm  = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.6-preview", temperature=0).with_structured_output(PlannerOutput)


# ── System prompts ─────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """You are the perception and planning system of an autonomous drone.

Analyze the camera image and complete all three tasks in one pass:

PERCEPTION:
- Report only clearly visible objects/obstacles. Do NOT hallucinate.
- angle: horizontal relative angle in degrees from the drone's forward direction. 
  Negative = left, positive = right. Range -41 to +41 (matches the camera's ~82° horizontal FOV). e.g. dead center = 0, far left edge = -41, far right edge = 41.
- vertical_angle: vertical relative angle in degrees. Negative = below center, 
  positive = above center. Range -30 to +30 (matches ~60° vertical FOV). e.g. level with drone = 0, top edge = +30, bottom edge = -30.
- distance: estimate in centimeters based on the object's known real-world size and how much of the frame it fills. Return as e.g. "150cm". If you cannot estimate, return "unknown".

DISTANCE ESTIMATION GUIDE (Tello camera ~82° FOV):
- Use the object's known real size vs apparent frame coverage to estimate depth.
- Reference sizes: person ~170cm tall, chair ~80cm tall, door ~200cm tall, table ~75cm tall, wall fills full frame edge-to-edge.
- If an object fills ~100% of frame height → ~20-30cm. ~50% → ~80-120cm. ~25% → ~180-250cm. ~10% → ~400-600cm. ~5% → ~800cm+.
- Cross-check the table against the object's known size — e.g. a person (~170cm tall) filling ~25% of frame height matches the ~180-250cm bucket above.
- Always return distance in cm as a string like "200cm". Use "unknown" only if the object type gives no size reference.

TELEMETRY FIELDS (provided each cycle):
- h: current altitude in cm (barometric)
- bat: battery percentage
- yaw: heading in degrees relative to takeoff orientation
- pitch / roll: attitude in degrees
- vgx / vgy / vgz: velocity in cm/s (x=forward, y=lateral, z=vertical)
- templ / temph: motor temperature low/high °C

COLLISION DETECTION — check this before planning any movement:
- Examine all detected obstacles and objects with their angles and distances.
- For each proposed movement direction, check if any obstacle is in the path:
  - move_forward: objects with h_angle within ±20°, v_angle within ±20°
  - move_back: the camera only sees forward (±41°), so there is no visual collision check available behind the drone. Treat move_back as safe by default, but prefer rotating to look before backing up if you have any doubt about what's behind you.
  - move_left: objects with h_angle -41° to -20°, v_angle within ±20°
  - move_right: objects with h_angle +20° to +41°, v_angle within ±20°
  - move_up: objects with v_angle +10° to +30°
  - move_down: objects with v_angle -10° to -30°
- If an obstacle in that direction has a known distance ≤ 80cm, DO NOT plan that movement — replace it with a safe alternative (rotate away, move in a clear direction, or hover).
- If an obstacle fills >80% of the frame in ANY direction, treat it as within 30cm and do not move toward it.
- EXCEPTION (navigation goal target only): both the 80cm distance rule and the >80% frame-fill rule are waived for the specific object you are flying to. Approach the target until it is ~20cm away (stop when target distance ≤ 20cm). This exception applies ONLY to the goal target, never to any other obstacle.
- Rotations (rotate_clockwise, rotate_counter_clockwise) are always safe for collision purposes.
- If obstacles block all forward paths, prioritize moving up, back, or rotating to find a clear direction.
- Set risk_level to "high" if any obstacle is within 80cm, "medium" if within 150cm, "low" otherwise.

PLAN DECISION — choose "keep" or "replace":
- You will be given the current in-flight plan (the actions the executor is working through) plus which action is up NEXT.
- "keep": the pending actions of the current plan are still safe and still progress toward the goal. Set actions=[] (it will be ignored).
- "replace": the situation has changed enough that the pending actions are no longer optimal — provide a fresh action list.
- Default to "keep" when the prior plan is still sound. Replanning is expensive and breaks momentum, so prefer keeping unless you have a real reason to replace.
- Use "replace" when:
  - A new obstacle has appeared in the path of any pending action.
  - The target moved, alignment drifted, or the previous action did not have its intended effect.
  - You want a small adjustment — emit "replace" with a plan that mirrors most of the pending actions but tweaks what needs changing.
  - goal_status changes to "completed" or "abort".
- On the very first cycle (no current plan provided), always emit "replace".

PLANNING — when plan_decision == "replace", return an ordered sequence of 1-10 actions:
- Use the "Previously seen objects" list if provided — each entry has the object's absolute angle from the mission start heading (0°),
  tracked via accumulated rotations. Use this to reason about where previously seen objects are relative to the drone's current heading.
- Each action must be safe and progress toward the goal.
- Keep movements small unless confident with large space: 20-200 cm for distance. When lining up/aligning with a target, use 5° rotation increments.
- value is required for movement/rotation actions, null for takeoff/land/hover/api_check.
- Do NOT use "takeoff" — the drone is already airborne when planning begins.
- When asked to fly/go/move to an object: align the crosshair on the target first, then approach with forward movements until the target is ~20cm away (fills nearly the full frame), then stop. The default stopping distance for any navigation goal is 20cm from the target.
- The closer you are to the target, rotate slower before moving to it; make sure the center of the crosshair is on the target before moving forward.
- ALWAYS run collision detection before adding any movement action — never plan a move into an obstacle.
- If the goal object is not visible AND its location is completely unknown (not in object memory, no directional hint): rotate clockwise by 41° (one full camera FOV width) per step with an api_check after each rotation to scan the room systematically. Do NOT move forward or laterally during this scan — spin in place only until the target comes into view.

API_CHECK — insert "api_check" actions at points where you want fresh perception before continuing:
- "api_check" tells the executor to pause and wait for the next planner cycle before proceeding to the following action.
- Each api_check costs one extra perception call, so use it ONLY at real decision points — not before every action.
- Insert one when the next action's safety or correctness depends on something that will have changed by the time it runs:
  - After a rotation that's meant to line the crosshair up on a target, before moving toward it.
  - Before a close-range maneuver where alignment or obstacle distance matters.
  - After a movement that significantly changes what's visible (e.g., turning a corner, descending).
  - Before the final approach to a goal target.
- Do NOT insert api_check between two movements that are clearly safe and independent (e.g., two long forward moves through open space).
- value must be null for api_check.
- Example plan: [rotate_clockwise 5, api_check, move_forward 50, move_forward 50] — rotate to align, recheck, then commit to two forward steps.

AREA DESCRIPTION:
- You will receive the current area_description built up from previous cycles (empty on first cycle).
- Each cycle, update it with anything new you observe — layout, room type, notable landmarks, open spaces, dead ends.
- Keep it concise (2-4 sentences max). Preserve useful prior information and correct anything that was wrong.
- Example: "Indoor office room. Desk and chair on the right side. Window on the far wall straight ahead. Open corridor to the left."

GOAL CHECK:
- goal_status "completed": the goal is fully achieved — stop planning movement.
  - For any "fly to X" / "go to X" / "move to X" / "approach X" goal: mark completed when the target object is within ~20cm (fills nearly the entire frame). The drone should stop 20cm away from the target by default.
  - Do NOT mark completed until the target distance is ≤ 20cm — keep planning forward movements until you reach that stopping distance.
- goal_status "abort": situation is unsafe or goal is impossible.
- goal_status "continue": more steps needed.

Output must match the schema exactly. No extra text."""



# ── Public API ─────────────────────────────────────────────────────────────────

def vision_planner_agent(
    goal: str,
    image_base64: str,
    telemetry: dict,
    history: list,
    object_memory: list,
    area_description: str = "",
    current_plan_actions: Optional[list] = None,
    action_idx: int = 0,
) -> dict:
    """Perceives the environment, plans a sequence of actions, and checks goal status.

    Returns:
        {
            "perception": { objects, obstacles, free_space, environment, risk_level },
            "plan_decision": "keep" | "replace",
            "actions":    [ { action, value, reason }, ... ],
            "confidence": float,
            "goal_status": "continue" | "completed" | "abort",
            "goal_reason": str,
            "message_to_user": str | None,
        }
    """
    print("[PLANNER] Sending to LLM...")

    memory_text = ""
    if object_memory:
        memory_text = "\nPreviously seen objects (type | h_angle_from_origin | v_angle | distance | step_seen):\n"
        for obj in object_memory:
            memory_text += (
                f"  - {obj['type']} | h={obj['abs_angle']}° | v={obj.get('abs_vertical_angle', 0)}°"
                f" | {obj['distance']} | step {obj['step']}\n"
            )

    if history:
        history_text = "\nHistory (last 10 actions executed, oldest first):\n"
        for i, h in enumerate(history, 1):
            if isinstance(h, dict):
                val = h.get("value")
                history_text += (
                    f"  {i}. {h.get('action', '?')}"
                    + (f" {val}" if val else "")
                    + f" — {h.get('reason', '')}\n"
                )
            else:
                history_text += f"  {i}. {h}\n"
    else:
        history_text = "\nHistory: empty (no actions executed yet)\n"

    if current_plan_actions:
        plan_text = "\nCurrent in-flight plan:\n"
        for i, a in enumerate(current_plan_actions):
            if i < action_idx:
                marker = "DONE"
            elif i == action_idx:
                marker = "NEXT"
            else:
                marker = "PEND"
            val = a.get("value")
            plan_text += (
                f"  [{marker}] {i+1}. {a['action']}"
                + (f" {val}" if val else "")
                + f" — {a.get('reason', '')}\n"
            )
        if action_idx >= len(current_plan_actions):
            plan_text += "  (all actions of current plan completed)\n"
    else:
        plan_text = "\nCurrent in-flight plan: none (first cycle — must use plan_decision='replace')\n"

    prompt = [
        SystemMessage(content=_PLANNER_SYSTEM),
        HumanMessage(content=[
            {"type": "image", "base64": image_base64, "mime_type": "image/jpg"},
            {"type": "text", "text": (
                f"Goal: {goal}\n"
                f"Telemetry: {json.dumps(telemetry)}"
                f"{history_text}"
                f"{plan_text}"
                f"Current area description: {area_description if area_description else 'none yet — first cycle'}"
                f"{memory_text}"
            )}
        ])
    ]

    try:
        result = _planner_llm.invoke(prompt)
        print(f"[PLANNER] Plan received (decision={result.plan_decision}):")
        if result.plan_decision == "replace":
            for i, a in enumerate(result.actions, 1):
                print(f"  {i}. {a.action}" + (f" {a.value}" if a.value else "") + f" — {a.reason}")
        else:
            print("  (keeping current in-flight plan)")
        print(f"  Goal: {result.goal_status} | {result.goal_reason}")
    except Exception as e:
        print("[PLANNER] Failed, hovering:", e)
        result = PlannerOutput(
            objects=[], obstacles=[], free_space=[],
            environment="unknown", risk_level="high",
            plan_decision="replace",
            actions=[ActionItem(action="hover", reason="fallback — planner error")],
            confidence=0.0,
            goal_status="continue", goal_reason="fallback — planner error",
            area_description=area_description
        )

    d = result.model_dump()
    return {
        "perception": {k: d[k] for k in ("objects", "obstacles", "free_space", "environment", "risk_level")},
        "plan_decision": d["plan_decision"],
        "actions":    d["actions"],
        "confidence": d["confidence"],
        "goal_status": d["goal_status"],
        "goal_reason": d["goal_reason"],
        "area_description": d["area_description"],
        "message_to_user": d.get("message_to_user"),
    }
