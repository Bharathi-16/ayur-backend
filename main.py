# AyurParam AI — Production Entry Point
from app.main import app

if __name__ == "__main__":
    # This part only runs during local development
    app.run(debug=True, host="0.0.0.0", port=8080)
