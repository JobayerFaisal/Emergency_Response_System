import streamlit as st
import asyncio
import websockets
import json
import streamlit.components.v1 as components
import uuid

# ---------------------------------------------------------
# REAL Geolocation Component (Fully working)
# ---------------------------------------------------------
def get_geolocation():

    # Unique key for Streamlit session
    key = f"geo_{uuid.uuid4().hex}"

    # JS code: runs in browser → sends data to Streamlit via postMessage
    geoloc_html = f"""
    <script>
        const sendLocation = () => {{
            if (!navigator.geolocation) {{
                window.parent.postMessage({{"type": "geo", "key": "{key}", "error": "Geolocation not supported"}}, "*");
                return;
            }}

            navigator.geolocation.getCurrentPosition(
                (pos) => {{
                    const data = {{
                        lat: pos.coords.latitude,
                        lng: pos.coords.longitude
                    }};
                    window.parent.postMessage({{"type": "geo", "key": "{key}", "data": data}}, "*");
                }},
                (err) => {{
                    window.parent.postMessage({{"type": "geo", "key": "{key}", "error": err.message}}, "*");
                }}
            );
        }};

        sendLocation();
    </script>
    """

    # Inject JS into Streamlit page
    components.html(geoloc_html, height=0)

    # Check if JS has already sent back the data
    return st.session_state.get(key, None)


# ---------------------------------------------------------
# Capture JS → Streamlit incoming messages
# ---------------------------------------------------------
# This MUST be placed before get_geolocation() usage
if "geo_listener" not in st.session_state:

    st.session_state.geo_listener = True

    geolistener_js = """
    <script>
        window.addEventListener("message", (event) => {
            if (event.data && event.data.type === "geo") {
                // send data into Streamlit state
                window.parent.postMessage(
                    { type: "streamlit:setComponentValue", key: event.data.key, value: event.data },
                    "*"
                );
            }
        });
    </script>
    """

    # This loads the listener only once
    components.html(geolistener_js, height=0)



# ---------------------------------------------------------
# Helper: process Streamlit setComponentValue messages
# ---------------------------------------------------------
def handle_geo_updates():
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith("geo_") and isinstance(st.session_state[k], dict):
            data = st.session_state[k]

            # If JS returned data correctly
            if "data" in data:
                return data["data"]

            # If JS returned an error
            if "error" in data:
                return {"error": data["error"]}

    return None



# =========================================================
#  PAGE UI
# =========================================================

st.title("🚨 Responder Chat Assistant")


# -------------------------------
# Responder ID
# -------------------------------
responder_id = st.text_input("Responder ID", "resp-001")


# -------------------------------
# GPS Location
# -------------------------------
st.subheader("📍 Your Current Location")

raw_location = get_geolocation()
location = handle_geo_updates()

if location is None:
    st.info("Detecting GPS location… Please allow browser permission.")
elif "error" in location:
    st.error(f"Location Error: {location['error']}")
else:
    st.success(f"Location detected: {location['lat']}, {location['lng']}")


# -------------------------------
# Chat Message Storage
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------
# WebSocket Endpoint
# -------------------------------
WEBSOCKET_URL = f"ws://disaster_backend:8000/api/v1/chat/{responder_id}"


# -------------------------------
# Send message via WebSocket
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
# Chat input
# -------------------------------
user_msg = st.chat_input("Type your message")

if user_msg:
    st.session_state.messages.append(("user", user_msg))

    payload = {
        "message": user_msg,
        "location": location,  # {lat, lng}
        "responder_id": responder_id,
    }

    reply = asyncio.run(send_message(payload))
    st.session_state.messages.append(("assistant", reply))

    st.rerun()


# -------------------------------
# Display chat history
# -------------------------------
for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)
