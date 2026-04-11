import time
from typing import Any, Dict, TypedDict, Annotated, List, Optional, Literal
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import base64
import json
from pydantic import BaseModel, Field

import threading
import keyboard

# from IPython.display import Image, display
from tello_sdk_controls_dir.main import SDK

load_dotenv()


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
    
class UserIntent(BaseModel):
    intent: Literal["chat", "set_goal", "stop_drone", "get_status"]
    new_goal: Optional[str] = None
    response_to_user: str
    confidence: float = Field(..., ge=0.0, le=1.0)

class GoalCheckOutput(BaseModel):
    status: Literal["continue", "completed", "abort", "pause"]
    reason: str
    message_to_user: Optional[str] = None
    suggested_new_goal: Optional[str] = None
    


# SDK Object
sdk = SDK()

def keep_alive():
    while True:
        try:
            sdk.tello.send_keepalive()   # or sdk.tello.send_command("command") if you prefer
            # Some people also use: sdk.tello.send_rc_control(0, 0, 0, 0)  # neutral sticks
        except Exception as e:
            print(f"Keep-alive error: {e}")  # optional, helps debugging
        time.sleep(5)   # ← changed from 10 to 5 (most reliable)

threading.Thread(target=keep_alive, daemon=True).start()


base_llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

vision_llm = base_llm.with_structured_output(VisionOutput)
planner_llm = base_llm.with_structured_output(ActionOutput)
router_llm = base_llm.with_structured_output(UserIntent)
checker_llm = base_llm.with_structured_output(GoalCheckOutput)

class State(TypedDict):
    
    messages: Annotated[List[BaseMessage], add_messages]
    goal: Optional[str]
    drone_active: bool
    telemetry: Dict[str, Any]
    perception: Dict[str, Any]
    action: Dict[str, Any]
    history: List[str]
    before_action_image: str | None
    




def router_agent(state: State) -> State:
    """Handles user messages: normal chat or detect drone commands and set goal."""
    print("[ROUTER] Processing user input...")

    # last_message = state["messages"][-1].content if state["messages"] else ""

    prompt = [
        SystemMessage(content="""
You are a helpful drone assistant that can chat naturally or control a real Tello drone.

Your job:
- If the user is just chatting → respond friendly and set intent="chat"
- If the user asks the drone to do something (takeoff, land, move, hover, fly to somewhere, check something, etc.) 
  → set intent="set_goal", extract a clear short goal, and respond with confirmation.
- If user says "stop", "land now", "abort", "emergency" → intent="stop_drone"
- If user asks for status → intent="get_status"

Be concise, friendly, and safe. Never promise impossible actions.
        """),
        *state["messages"]  # pass full conversation history
    ]

    try:
        decision: UserIntent = router_llm.invoke(prompt)
        print(f"[ROUTER] Intent: {decision.intent} | Goal: {decision.new_goal}")
    except Exception as e:
        print("[ROUTER] LLM failed, fallback to chat:", e)
        decision = UserIntent(
            intent="chat",
            new_goal=None,
            response_to_user="Sorry, I didn't understand. Could you rephrase?",
            confidence=0.0
        )

    updates = {
        "messages": [AIMessage(content=decision.response_to_user)]
    }

    if decision.intent == "set_goal" and decision.new_goal:
        updates["goal"] = decision.new_goal
        updates["drone_active"] = True
        updates["history"] = []  # reset drone history for new goal
    elif decision.intent == "stop_drone":
        updates["goal"] = None
        updates["drone_active"] = False
        updates["messages"].append(AIMessage(content="Drone command stopped. Returning to chat mode."))
    elif decision.intent == "get_status":
        status_msg = f"Current goal: {state.get('goal') or 'None'}. Drone active: {state.get('drone_active', False)}"
        updates["messages"].append(AIMessage(content=status_msg))

    return updates

def vision_agent(state: State) -> State:
    print("[VISION] Starting vision processing...")
    
    image_base64 = sdk.TakePicture()

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
                "mime_type": "image/jpg",
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
        "telemetry": sdk.DroneSystemInformation(),
        "perception": perception.model_dump(),
        "before_action_image": image_base64
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
Move toward the user's goal safely.

---

CORE RULES:

1. SAFETY FIRST
- Never move toward obstacles.
- Avoid high-risk or unknown areas.
- If uncertain → choose a safe action (hover or rotate).

2. SHORT-TERM ACTIONS
- Output ONLY one action.
- Keep movements small (20-50 units).
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
        HumanMessage(content=f""" Goal: {state.get('goal')} Perception: {json.dumps(state.get('perception'), indent=2)} Telemetry: {json.dumps(state.get('telemetry'), indent=2)} History:
{state.get('history')}
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
    action_dict = state.get("action")
    action = action_dict.get('action', 'UNKNOWN')
    value = action_dict.get('value', None)

    print(f"\n[EXECUTOR] ───── Step {len(state.get('history')) + 1} ─────")
    print(f"[EXECUTOR] Chosen action: {action}")
    if value is not None:
        print(f"[EXECUTOR] Value: {value}")
    print(f"[EXECUTOR] Reason: {action_dict.get('reason', '—')}")
    print(f"[EXECUTOR] Confidence: {action_dict.get('confidence', '—')}")

    sdk.DroneFlightController(action_dict)

    state.get("history").append(action)
    print(f"[EXECUTOR] Action appended to history. New length: {len(state.get('history'))}")
    
    return state



def goal_checker(state: State) -> State:
    """Checks if the current goal is completed, should continue, or needs to abort."""
    print("[CHECKER] Evaluating goal progress...")

    if not state.get("drone_active") or not state.get("goal"):
        return {"drone_active": False}
    
    image_base64 = sdk.TakePicture()

    prompt = [
        SystemMessage(content="""
You are the goal supervisor for an autonomous drone.

Your job is to make practical, action-oriented decisions to complete the goal efficiently while maintaining reasonable safety.

You must choose ONE:
- "continue"  → default choice; keep progressing toward the goal
- "completed" → goal clearly achieved
- "abort"     → only if there is a clear and immediate danger or critical failure
- "pause"     → temporary uncertainty that prevents safe progress (rare)

Decision Guidelines:
- Prefer "continue" unless there is strong evidence to stop.
- Do NOT abort for minor risks, uncertainty, or imperfect perception.
- Temporary ambiguity, partial visibility, or small obstacles are NORMAL → continue.
- Only use "abort" for:
  - imminent collision with no recovery
  - critically low battery
  - loss of control or unstable flight
  - explicit user stop intent
- Use "pause" only if the drone is completely stuck or perception is unusable.

Goal Handling:
- Focus on whether the drone is making progress toward the goal.
- If progress is being made → continue.
- If the goal appears achieved based on perception → completed.

Tone:
- Be decisive, not overly cautious.
- Avoid unnecessary stopping.
- Assume the drone is capable of basic obstacle avoidance.

Output:
- status: one of ["continue", "completed", "abort", "pause"]
                      
- reason: short and concrete explanation
- optional message_to_user

        """),
        HumanMessage(content=[f"""
Current Goal: {state.get('goal')}
Drone Active: {state.get('drone_active')}
Latest Perception: {json.dumps(state.get('perception'), indent=2)}
Telemetry: {json.dumps(state.get('telemetry'), indent=2)}
Action History (last 10): {state.get('history', [])[-10:]}
        """,
        {"type": "text", "text": "Image from the drone before action."},
         {
                "type": "image",
                "base64": state.get('before_action_image'),
                "mime_type": "image/jpg",
            },
            {"type": "text", "text": "Image from the drone after action."},
         {
                "type": "image",
                "base64": image_base64,
                "mime_type": "image/jpg",
            },

        
        
        ]
        
        )
    ]

    try:
        check = checker_llm.invoke(prompt)
        print(f"[CHECKER] Status: {check.status} | Reason: {check.reason}")
    except Exception as e:
        print("[CHECKER] Failed, default continue:", e)
        check = GoalCheckOutput(status="continue", reason="parsing fallback", message_to_user=None)

    updates: Dict[str, Any] = {
        "messages": []
    }

    if check.message_to_user:
        updates["messages"].append(AIMessage(content=check.message_to_user))

    if check.status in ["completed", "abort"]:
        updates["drone_active"] = False
        updates["goal"] = None
        if check.status == "completed":
            updates["messages"].append(AIMessage(content=f"Goal completed: {state.get('goal')}"))
        else:
            updates["messages"].append(AIMessage(content=f"Autonomous flight aborted: {check.reason}"))
    elif check.status == "pause":
        updates["drone_active"] = False  # pause until user resumes

    if check.suggested_new_goal:
        updates["goal"] = check.suggested_new_goal

    return updates



def route_after_router(state: State):
    """Decide what happens after router processes user input."""
    if state.get("drone_active") and state.get("goal"):
        return "vision_agent"      # start/continue drone loop
    else:
        return "end"                 # stay in chat mode

def route_after_checker(state: State):
    """After checker runs, decide next step."""
    status = state.get("drone_active", False)
    if status:
        return "vision_agent"      # continue the autonomous loop
    else:
        return "end"     # go back to conversation mode

# ────────────────────────────────────────────────


graph = StateGraph(State)

# Add all nodes
graph.add_node("router_agent", router_agent)
graph.add_node("vision_agent", vision_agent)
graph.add_node("planner_agent", planner_agent)
graph.add_node("executor", executor_node)
graph.add_node("goal_checker", goal_checker)

# Entry point is always the router (listens to user)
graph.set_entry_point("router_agent")

# Router decides whether to go to drone loop or end
graph.add_conditional_edges(
    "router_agent",
    route_after_router,
    {
        "vision_agent": "vision_agent",
        "end": END
    }
)

# Drone control loop
graph.add_edge("vision_agent", "planner_agent")
graph.add_edge("planner_agent", "executor")
graph.add_edge("executor", "goal_checker")

# Checker decides whether to loop back or return to conversation
graph.add_conditional_edges(
    "goal_checker",
    route_after_checker,
    {
        "vision_agent": "vision_agent",
        "end": END
    }
)

app = graph.compile()


#ai agent logic flow chart generator
# img_bytes = app.get_graph().draw_mermaid_png()

# with open("graph.png", "wb") as f:
#     f.write(img_bytes)
    
state = {
    "messages": [HumanMessage(content="Hi, I'm ready to fly the drone.")],
    "goal": None,
    "drone_active": False,
    "telemetry": {},
    "perception": {},
    "action": {},
    "history": []
}

print("\n===== Starting drone control loop =====")
print("Goal:", state.get("goal"))
print("Initial history:", state.get("history"))

# state = app.invoke(state)


# For continuous running with user input
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    new_state = {"messages": [HumanMessage(content=user_input)]}
    for chunk in app.stream(new_state, stream_mode="updates"):
        for node_name, node_updates in chunk.items():
            print(f"\n--- Update from node: {node_name} ---")
            if "messages" in node_updates:
                for msg in node_updates["messages"]:
                    if hasattr(msg, "pretty_print"):
                        msg.pretty_print()
                    else:
                        print(msg)
        
print("\n===== Loop finished =====")
print("Final history:", state.get("history"))