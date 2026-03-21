"""
Flask Application Factory
"""
import os
from flask import Flask, jsonify
from flask_cors import CORS

def create_app():
    app = Flask(__name__)

    # ── Config ──
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'ayurparam-dev-key-change-in-prod')
    app.config['DATABASE'] = os.path.join(os.path.dirname(__file__), '..', 'data', 'ayurparam.db')
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB upload limit

    # CORS configuration specifically for your Vercel frontend (Production + Previews)
    CORS(app, resources={r"/api/*": {"origins": [
        "https://ayur-web.vercel.app",
        "https://ayur-web-trivineai-3445s-projects.vercel.app",
        "https://ayur-web-git-main-trivineai-3445s-projects.vercel.app",
        "https://ayur-nmplwpzpb-trivineai-3445s-projects.vercel.app",
        "http://localhost:3000"
    ]}})



    # ── Database ──
    try:
        from app.models.database import init_db
        init_db(app)
    except ImportError:
        # If DB logic is missing or broken, add dummy init here or handle gracefully
        pass

    # ── Register Blueprints ──
    try:
        from app.routes.chat import chat_bp
        from app.routes.history import history_bp
        from app.routes.settings import settings_bp
        from app.routes.admin import admin_bp
        from app.routes.herbs import herbs_bp

        app.register_blueprint(chat_bp, url_prefix='/api')
        app.register_blueprint(history_bp, url_prefix='/api')
        app.register_blueprint(settings_bp, url_prefix='/api')
        app.register_blueprint(admin_bp, url_prefix='/api/admin')
        app.register_blueprint(herbs_bp, url_prefix='/api')
    except ImportError as e:
        print(f"Error loading routes: {e}")

    @app.route("/")
    def index():
        return jsonify({
            "name": "AyurParam API",
            "version": "1.0.0",
            "status": "online",
            "endpoints": [
                "/api/chat",
                "/api/history",
                "/api/herbs",
                "/api/settings",
                "/api/admin/health"
            ]
        })

    return app
