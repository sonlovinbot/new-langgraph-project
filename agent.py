import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# 1. Load biến môi trường (.env chứa OPENAI_API_KEY)
load_dotenv()

# 2. Khởi tạo LLM (GPT-4.1)
llm = init_chat_model("openai:gpt-4.1")

# 3. Định nghĩa trạng thái LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 4. Định nghĩa node "chatbot"
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# 5. Tạo đồ thị trạng thái
graph_builder = StateGraph(State)

# 6. Thêm node "chatbot"
graph_builder.add_node("chatbot", chatbot)

# ✅ 7. Thêm entry point → chạy từ START đến chatbot
graph_builder.add_edge(START, "chatbot")

# ✅ 8. Thêm exit point → kết thúc sau chatbot
graph_builder.add_edge("chatbot", END)

# ✅ 9. Compile đồ thị thành graph có thể gọi
graph = graph_builder.compile()

# 10. Chạy thử một lần
if __name__ == "__main__":
    user_input = input("Bạn: ")
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
    print("🤖 GPT-4.1:", result["messages"][-1].content)
