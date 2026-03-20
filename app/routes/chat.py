"""
Chat Routes — SSE streaming, stop, regenerate
"""
from flask import Blueprint, request, Response, stream_with_context, jsonify
from app.services.inference import generate_stream, stop_generation, get_status, PRESETS
from app.models.database import add_message, get_messages, delete_last_assistant_message, get_session

chat_bp = Blueprint('chat', __name__)


@chat_bp.route("/status")
def status():
    from app.services.inference import start_model_loading, model_state
    if not model_state["loaded"] and not model_state["loading"]:
        start_model_loading()
    return jsonify(get_status())

@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    session_id = data.get("session_id")
    history = data.get("history", [])
    config = data.get("config", {})
    system_prompt = data.get("system_prompt")

    # Apply preset if specified
    preset = data.get("preset")
    if preset and preset in PRESETS:
        from app.services.inference import PRESETS
        merged = PRESETS[preset].copy()
        merged.update(config)
        config = merged

    # Create session if it doesn't exist
    if not session_id:
        from app.models.database import create_session
        session_id = create_session(history[-1]["content"][:30] + "..." if history else "New Chat")

    # Save user message
    user_msg = history[-1]["content"] if history and history[-1]["role"] == "user" else ""
    if session_id and user_msg:
        add_message(session_id, "user", user_msg)

    def stream_and_save():
        full_response = ""
        token_count = 0
        latency_ms = 0

        for chunk in generate_stream(history, config, system_prompt):
            yield chunk
            # Parse the last chunk to get final stats
            if '"done": true' in chunk or '"stopped": true' in chunk:
                import json
                try:
                    payload = json.loads(chunk.replace("data: ", "").strip())
                    full_response = payload.get("full", "")
                    token_count = payload.get("token_count", 0)
                    latency_ms = payload.get("latency_ms", 0)
                    # Add session_id to the final payload if it was newly created
                    payload["session_id"] = session_id
                    # This is just for internal parsing, we yield the updated payload manually
                    # Wait, we need to yield it as well for the frontend
                    yield f"data: {json.dumps(payload)}\n\n"
                except:
                    pass

        # Save assistant response
        if session_id and full_response:
            add_message(session_id, "assistant", full_response, token_count, latency_ms)

    return Response(
        stream_with_context(stream_and_save()),
        content_type="text/event-stream"
    )


@chat_bp.route("/chat/stop", methods=["POST"])
def chat_stop():
    stop_generation()
    return jsonify({"status": "stopped"})


@chat_bp.route("/chat/regenerate", methods=["POST"])
def chat_regenerate():
    data = request.get_json()
    session_id = data.get("session_id")

    if session_id:
        delete_last_assistant_message(session_id)

    # Get updated history
    messages = get_messages(session_id) if session_id else data.get("history", [])
    history = [{"role": m["role"] if isinstance(m, dict) else m["role"],
                "content": m["content"] if isinstance(m, dict) else m["content"]}
               for m in messages]

    config = data.get("config", {})
    system_prompt = data.get("system_prompt")

    def stream_and_save():
        full_response = ""
        token_count = 0
        latency_ms = 0

        for chunk in generate_stream(history, config, system_prompt):
            yield chunk
            if '"done": true' in chunk:
                import json
                try:
                    payload = json.loads(chunk.replace("data: ", "").strip())
                    full_response = payload.get("full", "")
                    token_count = payload.get("token_count", 0)
                    latency_ms = payload.get("latency_ms", 0)
                except:
                    pass

        if session_id and full_response:
            add_message(session_id, "assistant", full_response, token_count, latency_ms)

    return Response(
        stream_with_context(stream_and_save()),
        content_type="text/event-stream"
    )
