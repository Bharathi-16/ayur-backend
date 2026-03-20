"""
Admin Routes — Health check, model reload, debug info
"""
import psutil
import torch
from flask import Blueprint, jsonify
from app.services.inference import get_status, reload_model, model_state

admin_bp = Blueprint('admin', __name__)


@admin_bp.route("/health")
def health():
    status = get_status()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    result = {
        "status": "healthy" if status["state"] == "ready" else "degraded",
        "model": status,
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "ram_total_gb": round(mem.total / (1024**3), 1),
            "ram_used_gb": round(mem.used / (1024**3), 1),
            "ram_percent": mem.percent,
            "disk_free_gb": round(disk.free / (1024**3), 1),
        }
    }

    if torch.cuda.is_available():
        result["system"]["gpu_name"] = torch.cuda.get_device_name(0)
        result["system"]["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
        result["system"]["vram_used_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 1)

    return jsonify(result)


@admin_bp.route("/reload", methods=["POST"])
def model_reload():
    reload_model()
    return jsonify({"status": "reloading"})


@admin_bp.route("/debug")
def debug_info():
    return jsonify({
        "model_state": {
            "loaded": model_state["loaded"],
            "device": model_state["device"],
            "model_id": model_state["model_id"],
            "error": model_state["error"],
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        },
        "system": {
            "python": __import__('sys').version,
            "platform": __import__('platform').platform(),
        }
    })
