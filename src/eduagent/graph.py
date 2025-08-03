from __future__ import annotations
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Union
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

load_dotenv()  # đảm bảo đọc OPENAI_API_KEY sớm


# ✨ Thêm OpenAI
from langchain_openai import ChatOpenAI

# === Reducer đúng chuẩn ===
def last_5_msgs(a: List[BaseMessage], b: List[BaseMessage]) -> List[BaseMessage]:
    return (a + b)[-5:]

@dataclass
class State:
    messages: Annotated[List[BaseMessage], last_5_msgs]


# === Agent Nodes ===

def planner_agent(state: State, config: RunnableConfig) -> dict:
    print("🔍 [Planner Agent] Suy nghĩ kế hoạch...")
    msg = AIMessage(content="Tôi đã hiểu yêu cầu. Để tôi lên kế hoạch cho bạn.")
    return {"messages": state.messages + [msg]}


# ✨ Gọi LLM thực tế từ OpenAI
llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.7)

def teacher_agent(state: State, config: RunnableConfig) -> dict:
    """Gọi GPT-4 để trả lời kiến thức cho giáo viên / học sinh."""
    print("📘 [Teacher Agent] Gọi OpenAI…")

    # Dùng toàn bộ lịch sử hội thoại làm ngữ cảnh
    response = llm.invoke(state.messages)

    # Nếu chỉ muốn dùng tin nhắn cuối cùng:
    #   response = llm.invoke(state.messages[-1].content)

    return {"messages": state.messages + [response]}


def parent_coach_agent(state: State, config: RunnableConfig) -> dict:
    print("👨‍👩‍👧 [Parent Coach Agent] Gợi ý cho phụ huynh...")
    msg = AIMessage(content="Gợi ý: hãy cùng con luyện tập 15 phút mỗi ngày và hỏi con xem con hiểu bài chưa.")
    return {"messages": state.messages + [msg]}


def rag_agent(state: State, config: RunnableConfig) -> dict:
    print("📚 [RAG Agent] Truy xuất thông tin giáo dục...")
    msg = AIMessage(content="Tôi đã tìm được tài liệu phù hợp với chủ đề bạn hỏi.")
    return {"messages": state.messages + [msg]}


def finish(state: State, config: RunnableConfig) -> dict:
    print("✅ [End] Kết thúc phiên trả lời.")
    return {"messages": state.messages}


# === Build Graph ===

graph = StateGraph(State)

graph.add_node("planner", planner_agent)
graph.add_node("teacher", teacher_agent)
graph.add_node("parent", parent_coach_agent)
graph.add_node("rag", rag_agent)
graph.add_node("end", finish)

graph.set_entry_point("planner")
graph.add_edge("planner", "teacher")
graph.add_edge("teacher", "rag")
graph.add_edge("rag", "parent")
graph.add_edge("parent", "end")

graph = graph.compile()
