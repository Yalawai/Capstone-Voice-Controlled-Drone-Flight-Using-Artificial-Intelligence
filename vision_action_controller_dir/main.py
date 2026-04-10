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
    status: Literal["continue", "completed", "abort"]
    reason: str
    message_to_user: Optional[str] = None


class CombinedOutput(BaseModel):
    # Perception
    objects: List[ObjectItem]
    obstacles: List[ObstacleItem]
    free_space: List[Literal["left", "center", "right"]]
    environment: Literal["indoor", "outdoor", "unknown"]
    risk_level: Literal["low", "medium", "high"]
    # Action
    action: Literal[
        "takeoff", "land", "hover",
        "move_forward", "move_back", "move_left", "move_right",
        "move_up", "move_down",
        "rotate_clockwise", "rotate_counter_clockwise"
    ]
    value: Optional[float]
    reason: str
    confidence: float
    # Goal check
    goal_status: Literal["continue", "completed", "abort"]
    goal_reason: str
    message_to_user: Optional[str] = None


base_llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
combined_llm = base_llm.with_structured_output(CombinedOutput)


COMBINED_SYSTEM = """You are the perception, control, and goal-supervision system of an autonomous drone.

Analyze the camera image and do all three tasks in one pass:

PERCEPTION:
- Report only clearly visible objects/obstacles. Do NOT hallucinate.
- direction: left/center/right (image thirds). distance: near/medium/far.

ACTION:
- Choose ONE safe action toward the goal. Keep movements small (20-50 units).
- SAFETY FIRST: never move toward obstacles. High risk_level → hover or rotate.
- value: required for movement/rotation, null otherwise.

GOAL CHECK:
- goal_status: continue / completed / abort.
- Be conservative: high risk_level → abort.

Output must match the schema exactly. No extra text."""


def vision_planner_agent(goal: str, image_base64: str, telemetry: dict, history: list) -> dict:
    """Analyzes image, plans next action, and checks goal status in a single LLM call.
    Returns {"perception": {...}, "action": {...}, "goal_check": GoalCheckOutput}
    """
    print("[VISION+PLANNER+CHECKER] Sending to LLM...")

    prompt = [
        SystemMessage(content=COMBINED_SYSTEM),
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
        result = combined_llm.invoke(prompt)
        print("[VISION+PLANNER+CHECKER] Result:")
        print(result.model_dump())
    except Exception as e:
        print("[VISION+PLANNER+CHECKER] Failed, fallback to hover:", e)
        result = CombinedOutput(
            objects=[], obstacles=[], free_space=[],
            environment="unknown", risk_level="high",
            action="hover", value=None,
            reason="fallback due to parsing error", confidence=0.0,
            goal_status="continue", goal_reason="fallback due to parsing error"
        )

    d = result.model_dump()
    perception_keys = ("objects", "obstacles", "free_space", "environment", "risk_level")
    action_keys = ("action", "value", "reason", "confidence")

    goal_check = GoalCheckOutput(
        status=d["goal_status"],
        reason=d["goal_reason"],
        message_to_user=d.get("message_to_user"),
    )

    return {
        "perception": {k: d[k] for k in perception_keys},
        "action": {k: d[k] for k in action_keys},
        "goal_check": goal_check,
    }