from flask import Flask
from flask_cors import CORS
from api.routes import api_bp
from config import Config
import os

def create_app():
    """
    Application factory function to create and configure the Flask app.
    
    Returns:
        Flask application
    """
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app, resources={r"/*": {"origins": "https://qstommyshu.github.io"}})
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", Config.SERVER_PORT))
    host = '0.0.0.0'
    debug = True
    
    app.run(host=host, port=port, debug=debug)

# if __name__ == '__main__':
#     app = create_app()
#     app.run(port=Config.SERVER_PORT, debug=True)
