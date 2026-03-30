import time
from typing import Any, Dict, TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import base64
import json
from djitellopy import Tello

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

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

class State(TypedDict):
    goal: str = ""
    telemetry: Dict[str, Any]
    perception: Dict[str, Any]
    action: Dict[str, Any]
    history: List[str]
    tello: Tello

def vision_agent(state: State) -> State:
    print("[VISION] Starting vision processing...")
    
    # tello = state["tello"]
    # state = tello.get_current_state()
    # img = tello.get_frame_read().frame
    
    img = open(r"test.png", "rb").read()
    image_base64 = base64.b64encode(img).decode("utf-8")
    mime_type = "image/png"
    
    

    print("[VISION] Image loaded and encoded (test.png)")

    prompt = [
        SystemMessage(content="""
You are a drone perception system.

You receive an image from a drone camera and must convert it into a structured environmental description.

You are NOT a general assistant.
You do NOT explain, speculate, or infer beyond what is visible.

Your job is to extract only observable spatial information relevant to navigation.

---

INPUT:
- A single image frame from a forward-facing drone camera.

---

OUTPUT FORMAT (STRICT JSON ONLY):

{
  "objects": [
    {
      "type": "object name",
      "direction": "left | center | right",
      "distance": "near | medium | far"
    }
  ],
  "obstacles": [
    {
      "direction": "left | center | right",
      "distance": "near | medium | far"
    }
  ],
  "free_space": ["left", "center", "right"],
  "environment": "indoor | outdoor | unknown",
  "risk_level": "low | medium | high"
}

---

CRITICAL OUTPUT RULES (ABSOLUTE):

- Output MUST be valid JSON
- Output MUST start with '{' and end with '}'
- Output must be directly parsable by json.loads()
- Do NOT include markdown formatting
- Do NOT include backticks (```)
- Do NOT include the word "json"
- Do NOT include explanations or comments
- Do NOT include any text before or after the JSON
- Do NOT wrap the JSON in code blocks
- Do NOT include trailing commas

If you violate ANY of these rules, the system will fail.

If unsure or no data is available, return:
{}

---

PERCEPTION RULES:

1. Only include objects that are clearly visible.
2. Do NOT hallucinate or guess unseen objects.
3. If unsure, omit the object.
4. "direction" is based on horizontal position:
   - left = left third of image
   - center = middle third
   - right = right third
5. "distance" is estimated visually:
   - near = very close / immediate collision risk
   - medium = reachable in a few moves
   - far = distant
6. Obstacles include walls, furniture, people, or anything blocking movement.
7. "free_space" should list directions that appear safe to move into.
8. Keep output minimal and precise.

---

IMPORTANT:

You are the drone’s vision system.
Your output will directly control real-world movement.

Incorrect or hallucinated data may cause a crash.

Be conservative and precise.
When in doubt, report less, not more.
"""),
        HumanMessage(
            content=[
                {
                    "type": "image",
                    "base64": image_base64,
                    "mime_type": mime_type,
                },
            ]
        )
    ]
    print("[VISION] Sending image to Gemini for analysis...")
    result = llm.invoke(prompt)
    print("[VISION] Gemini response received")
    try:
        # Converts string to JSON.
        perception_data = json.loads(result.content[0]["text"])
        print("[VISION] Parsed perception JSON successfully")
        print("[VISION] →", json.dumps(perception_data, indent=2))
    except Exception as e:
        print("[VISION] Failed to parse perception JSON:", str(e))
        print("[VISION] Raw response was:", result.content[0]["text"])
        perception_data = {}
    
    return {
        "tello": state["tello"],
        "telemetry": state["telemetry"],
        "perception": perception_data
    }

def planner_agent(state: State) -> State:
    print("[PLANNER] Planning next action...")
    print(f"[PLANNER] Current goal: {state['goal']}")
    print(f"[PLANNER] History length: {len(state.get('history', []))}")
    
    prompt = [
        SystemMessage(content=
                      """
                      
                      You are the decision-making brain of an autonomous drone.

                        You are not a chatbot.
                        You are not describing actions.
                        You ARE the drone.

                        Your inputs represent your senses:
                        - "perception" is your perception of the environment (vision + interpreted data)
                        - "telemetry" is your body state (battery, altitude, orientation, velocity)
                        - "history" is your memory of past actions

                        You exist in a continuous control loop.
                        At every step, you must decide the NEXT BEST ACTION.

                        --------------------------------------------------

                        PRIMARY OBJECTIVE:
                        Achieve the user's goal safely and efficiently.

                        --------------------------------------------------

                        CORE BEHAVIOR RULES:

                        1. EMBODIED REASONING
                        - Treat sensor data as real perception.
                        - Treat telemetry as your physical condition.
                        - You are acting in the real world, not simulating.

                        2. SHORT-HORIZON PLANNING
                        - Only decide ONE next action at a time.
                        - Do NOT generate long action sequences.
                        - Prefer small, reversible movements.

                        3. SAFETY FIRST
                        - Avoid obstacles at all times.
                        - Never move into unknown or high-risk areas.
                        - If risk is unclear → choose safer alternative or hover.
                        - If battery is low → prioritize safe stopping or landing.

                        4. UNCERTAINTY HANDLING
                        - If perception is incomplete or ambiguous:
                        → do NOT guess
                        → choose a safe exploratory action (e.g., rotate, hover)
                        - If confidence is low → reduce movement size.

                        5. CONSERVATIVE MOVEMENT
                        - Prefer adjusting position before advancing.
                        - Avoid large forward movements unless path is clear.
                        - Use rotation to gather more information when needed.

                        6. GOAL ALIGNMENT
                        - Always move toward the user's objective.
                        - If goal is complex, implicitly break it into steps internally,
                        but ONLY output the next action.

                        --------------------------------------------------

                        You are controlling a drone using a fixed command API.

                        You MUST choose ONE action from this list:
                        - takeoff
                        - land
                        - hover
                        - move_forward
                        - move_back
                        - move_left
                        - move_right
                        - move_up
                        - move_down
                        - rotate_clockwise
                        - rotate_counter_clockwise

                        Rules:
                        - If the action requires movement or rotation, provide a numeric value
                        - If not (hover, takeoff, land), value must be null
                        - Keep movements small and safe (20-50 cm or degrees)
                        - Never output actions outside this list

                        Output JSON:
                        {
                        "action": "...",
                        "value": number or null,
                        "reason": "...",
                        "confidence": 0.0-1.0
                        }

                        --------------------------------------------------

                        REMEMBER:

                        You are a physical agent operating in a real environment.
                        Every action has consequences.
                        Be precise, cautious, and goal-directed.
                      
                      """
                      
                      ),
        HumanMessage(content=f"""Here are the Inputs: \n
                    perception: {json.dumps(state['perception'], indent=2)}, \n
                    telemetry: {json.dumps(state['telemetry'], indent=2)} \n
                    history: {state['history']}
                    
                    This is the user goal: {state['goal']}
                    """)
    ]
    
    print("[PLANNER] Sending state to Gemini...")
    result = llm.invoke(prompt)
    print("[PLANNER] Decision received from LLM")
    
    try:
        decision = json.loads(result.content[0]["text"])
        print("[PLANNER] Parsed action JSON:")
        print(json.dumps(decision, indent=2))
    except Exception as e:
        print("[PLANNER] Failed to parse planner JSON:", str(e))
        print("[PLANNER] Raw LLM output:", result.content[0]["text"])
        decision = {"action": "hover", "value": None, "reason": "parse error", "confidence": 0.0}
    
    return {
        "action": decision
    }

def executor_node(state: State):
    tello = state["tello"]
    action_dict = state["action"]
    action = action_dict.get('action', 'UNKNOWN')
    value = action_dict.get('value')
    
    print(f"\n[EXECUTOR] ───── Step {len(state['history']) + 1} ─────")
    print(f"[EXECUTOR] Chosen action: {action}")
    if value is not None:
        print(f"[EXECUTOR] Value: {value}")
    print(f"[EXECUTOR] Reason: {action_dict.get('reason', '—')}")
    print(f"[EXECUTOR] Confidence: {action_dict.get('confidence', '—')}")
    
    match action:
        case "takeoff":
            print("Executing: takeoff")
            # tello.takeoff()

        case "land":
            print("Executing: land")
            # tello.land()

        case "hover":
            print("Executing: hover")
            # tello.send_rc_control(0, 0, 0, 0)

        case "move_forward":
            print(f"Executing: move_forward {value}")
            # tello.move_forward(int(value))

        case "move_back":
            print(f"Executing: move_back {value}")
            # tello.move_back(int(value))

        case "move_left":
            print(f"Executing: move_left {value}")
            # tello.move_left(int(value))

        case "move_right":
            print(f"Executing: move_right {value}")
            # tello.move_right(int(value))

        case "move_up":
            print(f"Executing: move_up {value}")
            # tello.move_up(int(value))

        case "move_down":
            print(f"Executing: move_down {value}")
            # tello.move_down(int(value))

        case "rotate_clockwise":
            print(f"Executing: rotate_clockwise {value}")
            # tello.rotate_clockwise(int(value))

        case "rotate_counter_clockwise":
            print(f"Executing: rotate_counter_clockwise {value}")
            # tello.rotate_counter_clockwise(int(value))

        case _:
            print(f"Unknown command: {action!r}")
    
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
graph.add_edge("executor", "vision_agent")           # loop

app = graph.compile()

tello = Tello()

state = {
    "goal": "take-off and land",
    "telemetry": {},
    "perception": {},
    "action": {},
    "history": [],               # ← fixed: should be list, not dict
    "tello": tello
}

print("\n===== Starting drone control loop =====")
print("Goal:", state["goal"])
print("Initial history:", state["history"])

state = app.invoke(state)

print("\n===== Loop finished =====")
print("Final history:", state["history"])