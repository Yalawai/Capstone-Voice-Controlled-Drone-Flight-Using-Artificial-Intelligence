from typing import Dict, List, Optional, Literal
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from pydantic import BaseModel

load_dotenv()


class ObjectItem(BaseModel):
    type: str
    direction: Literal["left", "center", "right"]
    distance: Literal["near", "medium", "far"]

class ObstacleItem(BaseModel):
    direction: Literal["left", "center", "right"]
    distance: Literal["near", "medium", "far"]

class VisionPlanOutput(BaseModel):
    objects: List[ObjectItem]
    obstacles: List[ObstacleItem]
    free_space: List[Literal["left", "center", "right"]]
    environment: Literal["indoor", "outdoor", "unknown"]
    risk_level: Literal["low", "medium", "high"]
    action: Literal[
        "takeoff", "land", "hover",
        "move_forward", "move_back", "move_left", "move_right",
        "move_up", "move_down",
        "rotate_clockwise", "rotate_counter_clockwise"
    ]
    value: Optional[float]
    reason: str
    confidence: float

class GoalCheckOutput(BaseModel):
    status: Literal["continue", "completed", "abort", "pause"]
    reason: str
    message_to_user: Optional[str] = None
    suggested_new_goal: Optional[str] = None


base_llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
vision_planner_llm = base_llm.with_structured_output(VisionPlanOutput)
checker_llm = base_llm.with_structured_output(GoalCheckOutput)


VISION_PLANNER_SYSTEM = """You are the perception and control system of an autonomous drone.

Analyze the camera image and choose the next action in one pass.

PERCEPTION:
- Report only clearly visible objects/obstacles. Do NOT hallucinate.
- direction: left/center/right (image thirds). distance: near/medium/far.

ACTION:
- Choose ONE safe action toward the goal. Keep movements small (20-50 units).
- SAFETY FIRST: never move toward obstacles. High risk_level → hover or rotate.
- value: required for movement/rotation, null otherwise.

Output must match the schema exactly. No extra text."""

CHECKER_SYSTEM = """You are the goal supervisor for an autonomous drone.
Decide: continue / completed / abort / pause.
Be conservative: high risk_level → abort or pause."""


def vision_planner_agent(goal: str, image_base64: str, telemetry: dict, history: list) -> dict:
    """Analyzes image and plans next action.
    Returns {"perception": {...}, "action": {...}}
    """
    print("[VISION+PLANNER] Sending to LLM...")

    prompt = [
        SystemMessage(content=VISION_PLANNER_SYSTEM),
        HumanMessage(content=[
            {
                "type": "image",
                "base64": image_base64,
                "mime_type": "image/jpg",
            },
            {
                "type": "text",
                "text": (
                    f"Goal: {goal}\n"
                    f"Telemetry: {json.dumps(telemetry)}\n"
                    f"History (last 10): {history[-10:]}"
                )
            }
        ])
    ]

    try:
        result = vision_planner_llm.invoke(prompt)
        print("[VISION+PLANNER] Result:")
        print(result.model_dump())
    except Exception as e:
        print("[VISION+PLANNER] Failed, fallback to hover:", e)
        result = VisionPlanOutput(
            objects=[], obstacles=[], free_space=[],
            environment="unknown", risk_level="high",
            action="hover", value=None,
            reason="fallback due to parsing error", confidence=0.0
        )

    d = result.model_dump()
    perception_keys = ("objects", "obstacles", "free_space", "environment", "risk_level")
    action_keys = ("action", "value", "reason", "confidence")

    return {
        "perception": {k: d[k] for k in perception_keys},
        "action": {k: d[k] for k in action_keys},
    }


def goal_checker(goal: str, perception: dict, telemetry: dict, history: list) -> GoalCheckOutput:
    """Checks whether the goal is done, should continue, or needs to abort."""
    print("[CHECKER] Evaluating goal progress...")

    prompt = [
        SystemMessage(content=CHECKER_SYSTEM),
        HumanMessage(content=(
            f"Goal: {goal}\n"
            f"Perception: {json.dumps(perception)}\n"
            f"Telemetry: {json.dumps(telemetry)}\n"
            f"History (last 10): {history[-10:]}"
        ))
    ]

    try:
        check = checker_llm.invoke(prompt)
        print(f"[CHECKER] Status: {check.status} | Reason: {check.reason}")
        return check
    except Exception as e:
        print("[CHECKER] Failed, default continue:", e)
        return GoalCheckOutput(status="continue", reason="parsing fallback")