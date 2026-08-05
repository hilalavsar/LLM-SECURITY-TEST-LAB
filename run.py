"""Dev entry point: python run.py  ->  http://127.0.0.1:5000"""

import os
import sys

# Make `app` importable no matter the working directory the server is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
