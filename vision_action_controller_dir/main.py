import time
from typing import Any, Dict, TypedDict, Annotated, List, Optional, Literal
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
import json
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import threading
import msvcrt


load_dotenv()


class ObjectItem(BaseModel):
    type: str
    direction: Literal["left", "center", "right"]
    distance: Literal["near", "medium", "far"]

class ObstacleItem(BaseModel):
    direction: Literal["left", "center", "right"]
    distance: Literal["near", "medium", "far"]

class VisionPlanOutput(BaseModel):
    """Combined perception + action output — single LLM call per control step."""
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

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    goal: Optional[str]
    drone_active: bool
    telemetry: Dict[str, Any]
    perception: Dict[str, Any]
    action: Dict[str, Any]
    proposed_action: Dict[str, Any]
    history: List[str]


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


def vision_planner_agent(state: State) -> State:
    print("[VISION+PLANNER] Fetching image and telemetry in parallel...")

    # Parallelize drone I/O — both are network calls to the drone
    with ThreadPoolExecutor(max_workers=2) as ex:
        picture_future = ex.submit(sdk.TakePicture)
        telemetry_future = ex.submit(sdk.DroneSystemInformation)
        image_base64 = picture_future.result()
        telemetry = telemetry_future.result()

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
                    f"Goal: {state.get('goal')}\n"
                    f"Telemetry: {json.dumps(telemetry)}\n"
                    f"History (last 10): {state.get('history', [])[-10:]}"
                )
            }
        ])
    ]

    print("[VISION+PLANNER] Sending to LLM...")
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
        "telemetry": telemetry,
        "perception": {k: d[k] for k in perception_keys},
        "action": {k: d[k] for k in action_keys},
    }


def executor_node(state: State):
    action_dict = state.get("action")
    action = action_dict.get("action", "UNKNOWN")
    value = action_dict.get("value", None)

    print(f"\n[EXECUTOR] ───── Step {len(state.get('history')) + 1} ─────")
    print(f"[EXECUTOR] Action: {action}" + (f" | Value: {value}" if value is not None else ""))
    print(f"[EXECUTOR] Reason: {action_dict.get('reason', '—')} | Confidence: {action_dict.get('confidence', '—')}")

    sdk.DroneFlightController(action_dict)

    state.get("history").append(action)
    print(f"[EXECUTOR] History length: {len(state.get('history'))}")

    return state


def goal_checker(state: State) -> State:
    """Checks if the current goal is completed, should continue, or needs to abort."""
    print("[CHECKER] Evaluating goal progress...")

    if not state.get("drone_active") or not state.get("goal"):
        return {"drone_active": False}

    prompt = [
        SystemMessage(content=CHECKER_SYSTEM),
        HumanMessage(content=(
            f"Goal: {state.get('goal')}\n"
            f"Perception: {json.dumps(state.get('perception'))}\n"
            f"Telemetry: {json.dumps(state.get('telemetry'))}\n"
            f"History (last 10): {state.get('history', [])[-10:]}"
        ))
    ]

    try:
        check = checker_llm.invoke(prompt)
        print(f"[CHECKER] Status: {check.status} | Reason: {check.reason}")
    except Exception as e:
        print("[CHECKER] Failed, default continue:", e)
        check = GoalCheckOutput(status="continue", reason="parsing fallback")

    updates: Dict[str, Any] = {"messages": []}

    if check.message_to_user:
        updates["messages"].append(AIMessage(content=check.message_to_user))

    if check.status in ["completed", "abort"]:
        updates["drone_active"] = False
        updates["goal"] = None
        if check.status == "completed":
            updates["messages"].append(AIMessage(content=f"Goal completed: {state.get('goal')}"))
        else:
            updates["messages"].append(AIMessage(content=f"Aborted: {check.reason}"))
    elif check.status == "pause":
        updates["drone_active"] = False

    if check.suggested_new_goal:
        updates["goal"] = check.suggested_new_goal

    return updates


def route_after_executor(state: State):
    """Run goal_checker every 3rd step to save an LLM call on most iterations."""
    step = len(state.get("history", []))
    if step % 3 == 0:
        return "goal_checker"
    return "vision_planner_agent"

def route_after_checker(state: State):
    if state.get("drone_active", False):
        return "vision_planner_agent"
    return "end"

def process_drone_cycle(user_input, image_base64, telemetry):
    global state

    state["messages"] = [HumanMessage(content=user_input)]
    state["voice_text"] = user_input
    state["latest_image"] = image_base64
    state["telemetry"] = telemetry

    state = app.invoke(state)

    return state.get("proposed_action", {})
# ────────────────────────────────────────────────

# ── Graph ──────────────────────────────────────────────────────────────────────

graph = StateGraph(State)

graph.add_node("vision_planner_agent", vision_planner_agent)
graph.add_node("executor", executor_node)
graph.add_node("goal_checker", goal_checker)

graph.set_entry_point("vision_planner_agent")
graph.add_edge("vision_planner_agent", "executor")

# Router decides whether to go to drone loop or end
graph.add_conditional_edges(
    "executor",
    route_after_executor,
    {
        "goal_checker": "goal_checker",
        "vision_planner_agent": "vision_planner_agent",
    }
)

graph.add_conditional_edges(
    "goal_checker",
    route_after_checker,
    {
        "vision_planner_agent": "vision_planner_agent",
        "end": END
    }
)

app = graph.compile()


print("\n===== Starting drone control loop =====")

while True:
    user_input = input("Goal: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    new_state = {
        "goal": user_input,
        "drone_active": True,
        "messages": [],
        "telemetry": {},
        "perception": {},
        "action": {},
        "history": []
    }
    for chunk in app.stream(new_state, stream_mode="updates"):
        print(chunk)

print("\n===== Loop finished =====")
