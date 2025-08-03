from __future__ import annotations
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Union
from dataclasses import dataclass, field

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

load_dotenv()  # đảm bảo đọc OPENAI_API_KEY sớm


# ✨ Thêm OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# Thử dùng vector store từ langchain_community; nếu không có thì sẽ fallback
try:  # pragma: no cover - phụ thuộc vào môi trường
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import FakeEmbeddings
except Exception:  # pragma: no cover - fallback nếu thiếu gói
    Chroma = None  # type: ignore
    FakeEmbeddings = None  # type: ignore

# === Reducer đúng chuẩn ===
def last_5_msgs(a: List[BaseMessage], b: List[BaseMessage]) -> List[BaseMessage]:
    return (a + b)[-5:]

@dataclass
class State:
    messages: Annotated[List[BaseMessage], last_5_msgs]
    memory: List[str] = field(default_factory=list)


# === Agent Nodes ===

def planner_agent(state: State, config: RunnableConfig) -> dict:
    print("🔍 [Planner Agent] Suy nghĩ kế hoạch...")
    msg = AIMessage(content="Tôi đã hiểu yêu cầu. Để tôi lên kế hoạch cho bạn.")
    return {
        "messages": state.messages + [msg],
        "memory": state.memory + ["Planner đã đưa ra kế hoạch."],
    }


# ✨ Gọi LLM thực tế từ OpenAI
llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.7)

def teacher_agent(state: State, config: RunnableConfig) -> dict:
    """Gọi GPT-4 để trả lời kiến thức cho giáo viên / học sinh."""
    print("📘 [Teacher Agent] Gọi OpenAI…")

    # Dùng toàn bộ lịch sử hội thoại làm ngữ cảnh
    try:
        response = llm.invoke(state.messages)
    except Exception:  # pragma: no cover - fallback khi không gọi được API
        response = AIMessage(content="(LLM không khả dụng, sử dụng trả lời mặc định.)")

    # Nếu chỉ muốn dùng tin nhắn cuối cùng:
    #   response = llm.invoke(state.messages[-1].content)

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


if Chroma and FakeEmbeddings:  # pragma: no cover
    # Dùng FakeEmbeddings để tránh phụ thuộc vào API ngoài khi chạy thử
    embeddings = FakeEmbeddings(size=512)

    # Một số tài liệu ví dụ trong cơ sở tri thức
    _docs = [
        Document(page_content="Toán học là nền tảng cho nhiều ngành khoa học khác."),
        Document(page_content="Khoa học lịch sử giúp học sinh hiểu về nguồn gốc dân tộc."),
    ]

    # Khởi tạo retriever sử dụng Chroma
    _retriever = Chroma.from_documents(_docs, embeddings).as_retriever()
else:  # Nếu thiếu thư viện, dùng retrieval đơn giản bằng tìm kiếm chuỗi
    _docs = [
        Document(page_content="Toán học là nền tảng cho nhiều ngành khoa học khác."),
        Document(page_content="Khoa học lịch sử giúp học sinh hiểu về nguồn gốc dân tộc."),
    ]

    def _retriever(query: str):  # type: ignore
        results = [
            d for d in _docs if any(word.lower() in d.page_content.lower() for word in query.split())
        ]
        return results or _docs


def rag_agent(state: State, config: RunnableConfig) -> dict:
    print("📚 [RAG Agent] Truy xuất thông tin giáo dục...")
    query = state.messages[-1].content
    if callable(getattr(_retriever, "invoke", None)):
        docs = _retriever.invoke(query)
    else:  # _retriever là hàm fallback
        docs = _retriever(query)  # type: ignore
    top_content = docs[0].page_content if docs else "Không tìm thấy tài liệu phù hợp."
    msg = AIMessage(
        content=f"Tôi đã tìm được tài liệu phù hợp: {top_content}",
    )
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
graph.add_node("rag", rag_agent)
graph.add_node("end", finish)

graph.set_entry_point("planner")
graph.add_edge("planner", "teacher")
graph.add_edge("teacher", "rag")
graph.add_edge("rag", "parent")
graph.add_edge("parent", "end")

graph = graph.compile()
