# backend/streamlit_app/pages/7_Responder_Chat.py

import streamlit as st
import asyncio
import websockets

st.title("🚨 Responder Chat Assistant")

responder_id = st.text_input("Responder ID", "resp-001")

if "messages" not in st.session_state:
    st.session_state.messages = []

async def chat_ws_send(msg: str):
    uri = f"ws://localhost:8000/api/v1/chat/{responder_id}"
    async with websockets.connect(uri) as ws:
        await ws.send(msg)
        reply = await ws.recv()
        return reply

user_msg = st.chat_input("Type your message")

if user_msg:
    st.session_state.messages.append(("user", user_msg))
    reply = asyncio.run(chat_ws_send(user_msg))
    st.session_state.messages.append(("assistant", reply))
    st.rerun()

for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)
