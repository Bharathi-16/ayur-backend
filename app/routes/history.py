"""
History Routes — Session CRUD, export
"""
import json
from flask import Blueprint, request, jsonify, Response
from app.models.database import (
    create_session, list_sessions, get_session,
    update_session_title, delete_session, get_messages
)

history_bp = Blueprint('history', __name__)


@history_bp.route("/sessions", methods=["GET"])
def sessions_list():
    return jsonify(list_sessions())


@history_bp.route("/sessions", methods=["POST"])
def sessions_create():
    data = request.get_json() or {}
    title = data.get("title", "New Chat")
    sid = create_session(title)
    return jsonify({"id": sid, "title": title})


@history_bp.route("/sessions/<sid>", methods=["GET"])
def session_get(sid):
    session = get_session(sid)
    if not session:
        return jsonify({"error": "Not found"}), 404
    messages = get_messages(sid)
    return jsonify({"session": session, "messages": messages})


@history_bp.route("/sessions/<sid>", methods=["PUT"])
def session_update(sid):
    data = request.get_json()
    title = data.get("title", "")
    if title:
        update_session_title(sid, title)
    return jsonify({"status": "updated"})


@history_bp.route("/sessions/<sid>", methods=["DELETE"])
def session_delete(sid):
    delete_session(sid)
    return jsonify({"status": "deleted"})


@history_bp.route("/sessions/<sid>/export", methods=["GET"])
def session_export(sid):
    fmt = request.args.get("format", "json")
    session = get_session(sid)
    if not session:
        return jsonify({"error": "Not found"}), 404

    messages = get_messages(sid)

    if fmt == "json":
        return jsonify({"session": session, "messages": messages})

    elif fmt == "txt":
        lines = [f"=== {session['title']} ===\n"]
        for m in messages:
            role = "You" if m["role"] == "user" else "AyurParam"
            lines.append(f"[{m['timestamp'][:19]}] {role}:\n{m['content']}\n")
        return Response("\n".join(lines), mimetype="text/plain",
                        headers={"Content-Disposition": f"attachment; filename={sid}.txt"})

    return jsonify({"error": "Unsupported format. Use json or txt"}), 400
