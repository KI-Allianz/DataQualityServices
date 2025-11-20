# app_factory.py
from flask import Flask, jsonify
from dotenv import load_dotenv
import logging,os

from .config import settings
from .routes import api_bp
from .ui.bp import ui_bp



def create_app() -> Flask:
    if os.getenv("FLASK_ENV") != "production":
        load_dotenv(override=False)
    app = Flask(__name__)

    
    secret = app.config.get("SECRET_KEY") or "change-me-in-prod"
    if not secret:
            raise RuntimeError("SECRET_KEY is missing in environment")
    app.secret_key = secret

    
    app.config.update(
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=True,
        PREFERRED_URL_SCHEME="https",
    )

    app.config["MAX_CONTENT_LENGTH"] = settings.max_json_mb * 1024 * 1024

    _configure_logging(app)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    # --- API (no auth) ---
    app.register_blueprint(api_bp, url_prefix="/api")

    # --- UI (no Keycloak / OAuth) ---
    #ui_bp = create_ui_blueprint()
    app.register_blueprint(ui_bp)  

    # Basic JSON errors for API callers
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"detail": "Bad request", "error": str(e)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"detail": "Not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        app.logger.exception("Unhandled exception: %s", e)
        return jsonify({"detail": "Internal server error"}), 500

    app.logger.info("Started %s v%s", settings.api_title, settings.api_version)
    return app


def _configure_logging(app: Flask):
    level_name = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level_name,
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    app.logger.setLevel(level_name)
