import time
from typing import TypedDict, Sequence, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import base64

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

SYSTEM_PROMPT = (("""
You are a drone vision system.

Analyse the image and return objects in a custom JSON format.

Important:
- Do NOT use keys such as "label", "labels", "box", "box_2d", "bbox", or "bounding_box"
- Do NOT output coordinates
- Convert every detected item into the exact schema below
- If any other output format would normally be used, ignore it and use this one instead

Rules:
- Only include clearly visible objects
- Distance must be based on object size in image (larger = nearer)
- Be consistent and deterministic
- Object names must be short labels only
- Each object name must be between 1 and 20 characters long
- Use lowercase letters, numbers, and underscores only
- Do not use spaces in object names
- If multiple objects of the same type exist, number them in order
- The first object should be named with suffix _1, the second _2, the third _3, and so on

Output exact JSON schema only:
[
  {"object": "chair_1", "distance": "near"},
  {"object": "chair_2", "distance": "medium"},
  {"object": "table_1", "distance": "far"}
]
"""))


image_bytes = open("roomtest.jpg", "rb").read()
image_base64 = base64.b64encode(image_bytes).decode("utf-8")
mime_type = "image/jpg"

message = HumanMessage(
    content=[
        {"type": "text", "text": "Detect objects and return structured output"},
        {
            "type": "image",
            "base64": image_base64,
            "mime_type": mime_type,
        },
    ]
)



response = model.invoke([message])

text = response.content[0]["text"].replace("\\n", "\n")
print(text)