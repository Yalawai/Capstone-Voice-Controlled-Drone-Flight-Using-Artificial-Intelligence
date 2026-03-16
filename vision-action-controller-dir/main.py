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

SYSTEM_PROMPT = ""

image_bytes = open("vision-action-controller-dir/flower.jpg", "rb").read()
image_base64 = base64.b64encode(image_bytes).decode("utf-8")
mime_type = "image/jpg"

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe the local image."},
        {
            "type": "image",
            "base64": image_base64,
            "mime_type": mime_type,
        },
    ]
)



response = model.invoke([message])

print(response)