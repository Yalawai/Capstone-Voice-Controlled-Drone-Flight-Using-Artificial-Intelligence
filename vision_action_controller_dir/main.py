import time
from typing import Any, Dict, TypedDict, Annotated, List, Optional, Literal
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import base64
import json
from pydantic import BaseModel, Field
from tello_sdk_controls_dir.main import SDK


load_dotenv()


ALLOWED_ACTIONS = {
    "takeoff",
    "land",
    "hover",
    "move_forward",
    "move_back",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "rotate_clockwise",
    "rotate_counter_clockwise"
}

class ObjectItem(BaseModel):
    type: str
    direction: Literal["left", "center", "right"]
    distance: Literal["near", "medium", "far"]

class ObstacleItem(BaseModel):
    direction: Literal["left", "center", "right"]
    distance: Literal["near", "medium", "far"]

class VisionOutput(BaseModel):
    objects: List[ObjectItem]
    obstacles: List[ObstacleItem]
    free_space: List[Literal["left", "center", "right"]]
    environment: Literal["indoor", "outdoor", "unknown"]
    risk_level: Literal["low", "medium", "high"]
    
class ActionOutput(BaseModel):
    action: Literal[
        "takeoff", "land", "hover",
        "move_forward", "move_back", "move_left", "move_right",
        "move_up", "move_down",
        "rotate_clockwise", "rotate_counter_clockwise"
    ]
    value: Optional[float]
    reason: str
    confidence: float

sdk = SDK()

base_llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

vision_llm = base_llm.with_structured_output(VisionOutput)
planner_llm = base_llm.with_structured_output(ActionOutput)

class State(TypedDict):
    goal: str = ""
    telemetry: Dict[str, Any]
    perception: Dict[str, Any]
    action: Dict[str, Any]
    history: List[str]
    drone_info: Dict[str, Any]


def vision_agent(state: State) -> State:
    print("[VISION] Starting vision processing...")
    sdk.TakePicture()
    #gets the image from the drone --- to do
    img = open(r"test.jpg", "rb").read()
    image_base64 = base64.b64encode(img).decode("utf-8")

    prompt = [
        SystemMessage(content="""
                      
                      You are the perception system of a drone.

Your job is to convert a single camera image into a structured description of the environment.

You must ONLY report what is clearly visible.
Do NOT guess, infer, or hallucinate.

---

RULES:

- Only include objects that are clearly visible.
- If unsure, omit the object.
- Do NOT invent objects or obstacles.
- Be conservative: reporting less is better than being wrong.

---

SPATIAL DEFINITIONS:

- direction:
  - "left" = left third of image
  - "center" = middle
  - "right" = right third

- distance:
  - "near" = immediate collision risk
  - "medium" = reachable in a few moves
  - "far" = distant

---

OUTPUT REQUIREMENTS:

- Output MUST match the schema exactly.
- No extra fields.
- No explanations.
- No text outside the structured output.

---

IMPORTANT:

Your output directly affects real-world movement.
Incorrect data may cause a crash.

Be precise and cautious.
                      
                      
                      
                      
                      """),
        HumanMessage(content=[
            {
                "type": "image",
                "base64": image_base64,
                "mime_type": "image/png",
            }
        ])
    ]

    print("[VISION] Sending to structured LLM...")
    
    try:
        perception = vision_llm.invoke(prompt)
        print("[VISION] Structured output received")
        print(perception.model_dump())
    except Exception as e:
        print("[VISION] Structured parsing failed:", e)
        perception = VisionOutput(
            objects=[],
            obstacles=[],
            free_space=[],
            environment="unknown",
            risk_level="high"
        )

    return {
        "drone_info": state["drone_info"],
            "telemetry": state["telemetry"],
        "perception": perception.model_dump()
    }

def planner_agent(state: State) -> State:
    print("[PLANNER] Planning next action...")

    prompt = [
        SystemMessage(content=""" 
                      
                      You are the control system of an autonomous drone.

You are NOT a chatbot.
You ARE the drone.

- Perception = your vision
- Telemetry = your body state
- Actions = real-world movement

You operate in a continuous control loop.
At each step, choose ONE safe and effective action.

---

PRIMARY OBJECTIVE:
Move toward the user’s goal safely.

---

CORE RULES:

1. SAFETY FIRST
- Never move toward obstacles.
- Avoid high-risk or unknown areas.
- If uncertain → choose a safe action (hover or rotate).

2. SHORT-TERM ACTIONS
- Output ONLY one action.
- Keep movements small (20–50 units).
- Prefer reversible actions.

3. CONSERVATIVE MOVEMENT
- Do not move forward unless path is clear.
- Use rotation to gather information if needed.

4. NO GUESSING
- Use ONLY the provided perception data.
- If data is unclear → act cautiously.

5. GOAL ALIGNMENT
- Always move toward the goal.
- If the goal is complex, handle it step-by-step internally.

---

ACTION RULES:

- Use ONLY allowed actions from the schema.
- If action requires movement or rotation → include value.
- If not → value must be null.

---

OUTPUT REQUIREMENTS:

- Must match the schema exactly.
- No extra text.
- No explanations outside fields.

---

IMPORTANT:

You are controlling a real drone.
Every action has physical consequences.

Be precise, cautious, and consistent.
                      
                      
                      """),
        HumanMessage(content=f""" Goal: {state['goal']} Perception: {json.dumps(state['perception'], indent=2)} Telemetry: {json.dumps(state['telemetry'], indent=2)} History:
{state['history']}
""")
    ]

    print("[PLANNER] Sending to structured LLM...")

    try:
        decision = planner_llm.invoke(prompt)
        print("[PLANNER] Structured action:")
        print(decision.model_dump())
    except Exception as e:
        print("[PLANNER] Failed, fallback to hover:", e)
        decision = ActionOutput(
            action="hover",
            value=None,
            reason="fallback due to parsing error",
            confidence=0.0
        )

    return {
        "action": decision.model_dump()
    }
    

def executor_node(state: State):
    tello = state["drone_info"]
    action_dict = state["action"]
    action = action_dict.get('action', 'UNKNOWN')
    value = action_dict.get('value', None)

    print(f"\n[EXECUTOR] ───── Step {len(state['history']) + 1} ─────")
    print(f"[EXECUTOR] Chosen action: {action}")
    if value is not None:
        print(f"[EXECUTOR] Value: {value}")
    print(f"[EXECUTOR] Reason: {action_dict.get('reason', '—')}")
    print(f"[EXECUTOR] Confidence: {action_dict.get('confidence', '—')}")

    sdk.DroneFlightController(action_dict)

    state["history"].append(action)
    print(f"[EXECUTOR] Action appended to history. New length: {len(state['history'])}")
    
    return state

# ────────────────────────────────────────────────

graph = StateGraph(State)

graph.add_node("vision_agent", vision_agent)
graph.add_node("planner_agent", planner_agent)
graph.add_node("executor", executor_node)

graph.set_entry_point("vision_agent")

graph.add_edge("vision_agent", "planner_agent")
graph.add_edge("planner_agent", "executor")
graph.add_edge("executor", "vision_agent")           

app = graph.compile()
Drone_info = sdk.DroneSystemInformation()

state = {
    "goal": "take-off and land",
    "telemetry": {},
    "perception": {},
    "action": {},
    "history": [],
    "drone_info": {}
}

print("\n===== Starting drone control loop =====")
print("Goal:", state["goal"])
print("Initial history:", state["history"])

state = app.invoke(state)

print("\n===== Loop finished =====")
print("Final history:", state["history"])