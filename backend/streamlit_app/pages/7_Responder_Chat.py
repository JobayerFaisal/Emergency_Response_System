# path: backend/streamlit_app/pages/7_Responder_Chat.py

import streamlit as st
import asyncio
import websockets

st.title("Responder Chat Agent")

responder_id = st.text_input("Responder ID", "resp-001")

if "chat" not in st.session_state:
    st.session_state.chat = []

async def chat_loop():
    uri = f"ws://localhost:8000/api/v1/chat/{responder_id}"

    async with websockets.connect(uri) as ws:
        while True:
            user_msg = st.chat_input("Message")
            if user_msg:
                st.session_state.chat.append(("user", user_msg))
                await ws.send(user_msg)

                bot_reply = await ws.recv()
                st.session_state.chat.append(("assistant", bot_reply))
                st.rerun()

for role, msg in st.session_state.chat:
    st.chat_message(role).write(msg)

asyncio.run(chat_loop())
