"""
AyurParam AI — Simple Developer Launcher
Run: python run.py
Web: http://localhost:8080
"""
from app.main import app

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🌿 AyurParam AI — Ayurveda Intelligence Platform")
    print("="*60)
    print("  Status:  http://localhost:8080/api/status")
    print("  Health:  http://localhost:8080/api/admin/health")
    print("="*60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=8080, threaded=True)
