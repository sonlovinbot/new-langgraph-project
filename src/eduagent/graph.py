from __future__ import annotations
import os
from dotenv import load_dotenv
from typing import Annotated, List, Optional
from dataclasses import dataclass, field

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

load_dotenv()  # đảm bảo đọc OPENAI_API_KEY sớm

# ✨ Thêm OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# Thử dùng vector store từ langchain_community; nếu không có thì sẽ fallback
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import FakeEmbeddings
except Exception:
    Chroma = None
    FakeEmbeddings = None

# === Reducer đúng chuẩn ===
def last_5_msgs(a: List[BaseMessage], b: List[BaseMessage]) -> List[BaseMessage]:
    return (a + b)[-5:]

@dataclass
class State:
    messages: Annotated[List[BaseMessage], last_5_msgs]
    memory: List[str] = field(default_factory=list)
    next_agent: Optional[str] = None

# ✨ Gọi LLM thực tế từ OpenAI
llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.7)

# === Agent Nodes ===

def planner_agent(state: State, config: RunnableConfig) -> dict:
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
    return {
        "messages": state.messages + [msg],
        "memory": state.memory + ["Planner đã đưa ra kế hoạch."],
        "next_agent": next_agent,
    }

def teacher_agent(state: State, config: RunnableConfig) -> dict:
    print("📘 [Teacher Agent] Gọi OpenAI…")
    try:
        response = llm.invoke(state.messages)
    except Exception:
        response = AIMessage(content="(LLM không khả dụng, sử dụng trả lời mặc định.)")
    return {
        "messages": state.messages + [response],
        "memory": state.memory + [f"Teacher trả lời: {response.content}"],
    }

def parent_coach_agent(state: State, config: RunnableConfig) -> dict:
    print("👨‍👩‍👧 [Parent Coach Agent] Gợi ý cho phụ huynh...")
    msg = AIMessage(content="Gợi ý: hãy cùng con luyện tập 15 phút mỗi ngày và hỏi con xem con hiểu bài chưa.")
    return {
        "messages": state.messages + [msg],
        "memory": state.memory + ["Parent Coach đã gợi ý luyện tập."],
    }

def visual_agent(state: State, config: RunnableConfig) -> dict:
    print("🖼️ [Visual Agent] Tạo nội dung trực quan...")
    msg = AIMessage(content="Đây là nội dung trực quan cho yêu cầu của bạn.")
    return {
        "messages": state.messages + [msg],
        "memory": state.memory + ["Visual đã tạo nội dung trực quan."],
    }

if Chroma and FakeEmbeddings:
    embeddings = FakeEmbeddings(size=512)
    _docs = [
        Document(page_content="Toán học là nền tảng cho nhiều ngành khoa học khác."),
        Document(page_content="Khoa học lịch sử giúp học sinh hiểu về nguồn gốc dân tộc."),
    ]
    _retriever = Chroma.from_documents(_docs, embeddings).as_retriever()
else:
    _docs = [
        Document(page_content="Toán học là nền tảng cho nhiều ngành khoa học khác."),
        Document(page_content="Khoa học lịch sử giúp học sinh hiểu về nguồn gốc dân tộc."),
    ]
    def _retriever(query: str):
        results = [d for d in _docs if any(word.lower() in d.page_content.lower() for word in query.split())]
        return results or _docs

def rag_agent(state: State, config: RunnableConfig) -> dict:
    print("📚 [RAG Agent] Truy xuất thông tin giáo dục...")
    query = state.messages[-1].content
    if callable(getattr(_retriever, "invoke", None)):
        docs = _retriever.invoke(query)
    else:
        docs = _retriever(query)
    top_content = docs[0].page_content if docs else "Không tìm thấy tài liệu phù hợp."
    msg = AIMessage(content=f"Tôi đã tìm được tài liệu phù hợp: {top_content}")
    return {
        "messages": state.messages + [msg],
        "memory": state.memory + [f"RAG kết quả: {top_content}"],
    }

def finish(state: State, config: RunnableConfig) -> dict:
    print("✅ [End] Kết thúc phiên trả lời.")
    return {"messages": state.messages, "memory": state.memory}

# === Build Graph ===

graph = StateGraph(State)

graph.add_node("planner", planner_agent)
graph.add_node("teacher", teacher_agent)
graph.add_node("parent", parent_coach_agent)
graph.add_node("visual", visual_agent)
graph.add_node("rag", rag_agent)
graph.add_node("end", finish)

graph.set_entry_point("planner")
graph.add_edge("planner", "teacher")
graph.add_edge("teacher", "rag")
graph.add_edge("rag", "parent")
graph.add_edge("parent", "end")

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
