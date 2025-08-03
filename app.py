import os
import sys
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Cho phép import từ thư mục src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from eduagent.graph import graph  # đã sửa từ 'agent.graph' sang đúng 'eduagent.graph'

load_dotenv()

st.title("🎓 Edu-Global AI Agent")

if "history" not in st.session_state:
    st.session_state.history = []
if "visuals" not in st.session_state:
    st.session_state.visuals = []

user_input = st.chat_input("Nhập câu hỏi…")
if user_input:
    st.session_state.history.append(HumanMessage(content=user_input))
    with st.spinner("Đang suy nghĩ…"):
        out = asyncio.run(
            graph.ainvoke(
                {
                    "messages": st.session_state.history,
                    "visuals": st.session_state.visuals,
                }
            )
        )
        st.session_state.history = out["messages"]  # ✅ không được thụt lề lệch
        st.session_state.visuals = out.get("visuals", [])

for m in st.session_state.history:
    role = "👤" if m.type == "human" else "🤖"
    st.markdown(f"**{role}**: {m.content}")

for img in st.session_state.visuals:
    st.image(img)
