"""
Settings Routes — Inference configuration
"""
import json
from flask import Blueprint, request, jsonify
from app.models.database import get_setting, set_setting
from app.services.inference import DEFAULT_CONFIG, PRESETS, SYSTEM_PROMPT

settings_bp = Blueprint('settings', __name__)


@settings_bp.route("/settings", methods=["GET"])
def get_settings():
    saved = get_setting("inference_config")
    config = json.loads(saved) if saved else DEFAULT_CONFIG.copy()
    system_prompt = get_setting("system_prompt") or SYSTEM_PROMPT
    return jsonify({
        "config": config,
        "system_prompt": system_prompt,
        "presets": list(PRESETS.keys()),
        "defaults": DEFAULT_CONFIG,
    })


@settings_bp.route("/settings", methods=["PUT"])
def update_settings():
    data = request.get_json()
    if "config" in data:
        set_setting("inference_config", json.dumps(data["config"]))
    if "system_prompt" in data:
        set_setting("system_prompt", data["system_prompt"])
    return jsonify({"status": "saved"})


@settings_bp.route("/settings/reset", methods=["POST"])
def reset_settings():
    set_setting("inference_config", json.dumps(DEFAULT_CONFIG))
    set_setting("system_prompt", SYSTEM_PROMPT)
    return jsonify({"status": "reset"})
