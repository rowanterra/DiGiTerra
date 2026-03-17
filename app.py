# Web entry point. All logic lives in core/, routes/, python_scripts/.
# Usage: python app.py   or   gunicorn app:app
from core.flask_app import app, create_app, run_app

if __name__ == "__main__":
    run_app(debug=False)
