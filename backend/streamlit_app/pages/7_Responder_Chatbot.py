# backend/streamlit_app/pages/7_Responder_Chat.py

import streamlit as st
import asyncio
import websockets
import json
import streamlit as st
import streamlit.components.v1 as components
import json
import uuid

# -------------------------------
def get_geolocation():
    geo_id = str(uuid.uuid4()).replace("-", "")

    html_code = f"""
    <script>
    const sendLocation = () => {{
        const elem = window.parent.document.getElementById("{geo_id}");
        if (navigator.geolocation) {{
            navigator.geolocation.getCurrentPosition(
                (pos) => {{
                    const data = {{
                        lat: pos.coords.latitude,
                        lng: pos.coords.longitude
                    }};
                    elem.value = JSON.stringify(data);
                    elem.dispatchEvent(new Event("change", {{ bubbles: true }}));
                }},
                (err) => {{
                    elem.value = JSON.stringify({{"error": err.message}});
                    elem.dispatchEvent(new Event("change", {{ bubbles: true }}));
                }}
            );
        }} else {{
            elem.value = JSON.stringify({{"error": "Geolocation not supported"}});
            elem.dispatchEvent(new Event("change", {{ bubbles: true }}));
        }}
    }};
    sendLocation();
    </script>

    <input type="hidden" id="{geo_id}" />
    """

    # Inject JS
    components.html(html_code, height=0)

    # Read the value after JS writes it back
    if geo_id in st.session_state:
        try:
            return json.loads(st.session_state[geo_id])
        except:
            return None
    else:
        # Prepare a listener for when HTML sets the value
        st.session_state[geo_id] = "{}"
        return None












st.title("🚨 Responder Chat Assistant")

# -------------------------------
# Responder ID (Static or Dynamic)
# -------------------------------
responder_id = st.text_input("Responder ID", "resp-001")

# -------------------------------
# Load GPS Location
# -------------------------------
st.subheader("📍 Your Current Location")

location = get_geolocation()   # This triggers the browser's GPS permission popup

if location:
    st.success(f"Location detected: {location['lat']}, {location['lng']}")
else:
    st.warning("Location not detected yet. Please allow browser permission.")

# -------------------------------
# Chat Message Storage
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# WebSocket Endpoint
# -------------------------------
WEBSOCKET_URL = f"ws://localhost:8000/api/v1/chat/{responder_id}"


# -------------------------------
# WebSocket Message Sender
# -------------------------------
async def send_message(payload: dict):
    try:
        async with websockets.connect(WEBSOCKET_URL) as ws:
            await ws.send(json.dumps(payload))
            reply = await ws.recv()
            return reply
    except Exception as e:
        return f"[Connection Error] {e}"


# -------------------------------
# Chat Input Section
# -------------------------------
user_msg = st.chat_input("Type your message")

if user_msg:
    # Save user message
    st.session_state.messages.append(("user", user_msg))

    # Prepare payload for backend
    payload = {
        "message": user_msg,
        "location": location,  # {"lat": .., "lng": ..}
        "responder_id": responder_id,
    }

    # Send message & get reply
    reply = asyncio.run(send_message(payload))

    # Save assistant reply
    st.session_state.messages.append(("assistant", reply))

    st.rerun()


# -------------------------------
# Display Chat History
# -------------------------------
for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)
