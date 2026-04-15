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
    direction: Literal["left", "center", "right"]
    distance: Literal["near", "medium", "far"]

class ObstacleItem(BaseModel):
    direction: Literal["left", "center", "right"]
    distance: Literal["near", "medium", "far"]

_ACTION_LITERAL = Literal[
    "takeoff", "land", "hover",
    "move_forward", "move_back", "move_left", "move_right",
    "move_up", "move_down",
    "rotate_clockwise", "rotate_counter_clockwise"
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
    actions: List[ActionItem]       # ordered sequence of 1-5 actions
    confidence: float
    goal_status: Literal["continue", "completed", "abort"]
    goal_reason: str
    message_to_user: Optional[str] = None


# ── Avoidance output ───────────────────────────────────────────────────────────

class AvoidanceOutput(BaseModel):
    safe: bool      # True = safe to execute, False = skip this action
    reason: str


# ── LLMs ──────────────────────────────────────────────────────────────────────

_planner_llm  = ChatGoogleGenerativeAI(model="gemini-3.1-pro", temperature=0).with_structured_output(PlannerOutput)
_avoidance_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, thinking_budget=0).with_structured_output(AvoidanceOutput)


# ── System prompts ─────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """You are the perception and planning system of an autonomous drone.

Analyze the camera image and complete all three tasks in one pass:

PERCEPTION:
- Report only clearly visible objects/obstacles. Do NOT hallucinate.
- direction: left/center/right (image thirds). distance: near/medium/far.

PLANNING — return an ordered sequence of 1-5 actions:
- Each action must be safe and progress toward the goal.
- Keep movements small: 20-50 cm for distance, 20-90 degrees for rotation.
- value is required for movement/rotation actions, null for takeoff/land/hover.
- If the drone has not taken off yet, the first action must be "takeoff".

GOAL CHECK:
- goal_status "completed": the goal is fully achieved — stop planning movement.
- goal_status "abort": situation is unsafe or goal is impossible.
- goal_status "continue": more steps needed.
- Be conservative: high risk_level → abort.

Output must match the schema exactly. No extra text."""

_AVOIDANCE_SYSTEM = """You are the obstacle avoidance safety system of an autonomous drone.

You receive the current camera image and a proposed action.
Respond with safe=true if it is safe to execute, or safe=false if it is not.

RULES:
- safe=false imminent danger hitting object or environment.
- safe=true no imminent danger hitting object or environment .

Output must match the schema exactly. No extra text."""

# Movement actions that require a valid value
_MOVEMENT_ACTIONS = frozenset({
    "move_forward", "move_back", "move_left", "move_right",
    "move_up", "move_down", "rotate_clockwise", "rotate_counter_clockwise"
})


# ── Public API ─────────────────────────────────────────────────────────────────

def vision_planner_agent(goal: str, image_base64: str, telemetry: dict, history: list) -> dict:
    """Perceives the environment, plans a sequence of actions, and checks goal status.

    Returns:
        {
            "perception": { objects, obstacles, free_space, environment, risk_level },
            "actions":    [ { action, value, reason }, ... ],
            "confidence": float,
            "goal_status": "continue" | "completed" | "abort",
            "goal_reason": str,
            "message_to_user": str | None,
        }
    """
    print("[PLANNER] Sending to LLM...")

    prompt = [
        SystemMessage(content=_PLANNER_SYSTEM),
        HumanMessage(content=[
            {"type": "image", "base64": image_base64, "mime_type": "image/jpg"},
            {"type": "text", "text": (
                f"Goal: {goal}\n"
                f"Telemetry: {json.dumps(telemetry)}\n"
                f"History (last 10): {list(history)}"
            )}
        ])
    ]

    try:
        result = _planner_llm.invoke(prompt)
        print("[PLANNER] Plan received:")
        for i, a in enumerate(result.actions, 1):
            print(f"  {i}. {a.action}" + (f" {a.value}" if a.value else "") + f" — {a.reason}")
        print(f"  Goal: {result.goal_status} | {result.goal_reason}")
    except Exception as e:
        print("[PLANNER] Failed, hovering:", e)
        result = PlannerOutput(
            objects=[], obstacles=[], free_space=[],
            environment="unknown", risk_level="high",
            actions=[ActionItem(action="hover", reason="fallback — planner error")],
            confidence=0.0,
            goal_status="continue", goal_reason="fallback — planner error"
        )

    d = result.model_dump()
    return {
        "perception": {k: d[k] for k in ("objects", "obstacles", "free_space", "environment", "risk_level")},
        "actions":    d["actions"],
        "confidence": d["confidence"],
        "goal_status": d["goal_status"],
        "goal_reason": d["goal_reason"],
        "message_to_user": d.get("message_to_user"),
    }


def object_avoidance_agent(proposed_action: dict, image_base64: str) -> dict:
    """Checks whether a proposed action is safe given the current camera image.

    Skips the LLM check for non-movement actions (takeoff, land, hover) and
    approves them immediately.

    Returns AvoidanceOutput as a dict:
        { "safe": bool, "reason": str }
    """
    action_name = proposed_action["action"]

    # Non-movement actions don't need obstacle checking
    if action_name not in _MOVEMENT_ACTIONS:
        return {
            "safe": True,
            "reason": "non-movement action, no avoidance check needed"
        }

    print(f"[AVOIDANCE] Checking: {action_name} value={proposed_action.get('value')}")

    prompt = [
        SystemMessage(content=_AVOIDANCE_SYSTEM),
        HumanMessage(content=[
            {"type": "image", "base64": image_base64, "mime_type": "image/jpg"},
            {"type": "text", "text": (
                f"Proposed action: {action_name}\n"
                f"Value: {proposed_action.get('value')}\n"
                f"Reason: {proposed_action.get('reason', '')}"
            )}
        ])
    ]

    try:
        result = _avoidance_llm.invoke(prompt)
        print(f"[AVOIDANCE] {'SAFE' if result.safe else 'UNSAFE'}: {result.reason}")
        return result.model_dump()
    except Exception as e:
        print("[AVOIDANCE] Failed, blocking for safety:", e)
        return {"safe": False, "reason": f"avoidance check error: {e}"}
