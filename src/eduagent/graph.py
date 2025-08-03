from __future__ import annotations
from dotenv import load_dotenv
from typing import Annotated, List, Optional
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph

load_dotenv()  # đảm bảo đọc OPENAI_API_KEY sớm


# ✨ Thêm OpenAI
from langchain_openai import ChatOpenAI

# === Reducer đúng chuẩn ===
def last_5_msgs(a: List[BaseMessage], b: List[BaseMessage]) -> List[BaseMessage]:
    return (a + b)[-5:]

@dataclass
class State:
    messages: Annotated[List[BaseMessage], last_5_msgs]
    next_agent: Optional[str] = None


# === Agent Nodes ===

def planner_agent(state: State, config: RunnableConfig) -> dict:
    """Sử dụng LLM để suy luận bước tiếp theo và chọn agent phù hợp."""
    print("🔍 [Planner Agent] Suy nghĩ kế hoạch...")

    planning_prompt = state.messages + [
        HumanMessage(
            content=(
                "Dựa vào hội thoại trên, hãy lên kế hoạch bước tiếp theo cho hệ thống. "
                "Chọn một trong các agent sau để xử lý: visual, teacher, rag. "
                "Trả lời duy nhất bằng JSON với hai khóa 'plan' và 'next_agent'."
            )
        )
    ]

    response = llm.invoke(planning_prompt)

    try:
        import json

        parsed = json.loads(response.content)
        plan_text = parsed.get("plan", "")
        next_agent = parsed.get("next_agent", "teacher")
    except Exception:
        plan_text = response.content
        next_agent = "teacher"

    msg = AIMessage(content=plan_text)
    return {"messages": state.messages + [msg], "next_agent": next_agent}


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


def visual_agent(state: State, config: RunnableConfig) -> dict:
    print("🖼️ [Visual Agent] Tạo nội dung trực quan...")
    msg = AIMessage(content="Đây là nội dung trực quan cho yêu cầu của bạn.")
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
graph.add_node("visual", visual_agent)
graph.add_node("rag", rag_agent)
graph.add_node("end", finish)

graph.set_entry_point("planner")
graph.add_conditional_edges(
    "planner",
    lambda state: state.next_agent,
    {
        "teacher": "teacher",
        "visual": "visual",
        "rag": "rag",
    },
)
graph.add_edge("teacher", "end")
graph.add_edge("visual", "end")
graph.add_edge("rag", "end")

graph = graph.compile()
