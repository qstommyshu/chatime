from flask import Flask
from flask_cors import CORS
from api.routes import api_bp
from config import Config

def create_app():
    """
    Application factory function to create and configure the Flask app.
    
    Returns:
        Flask application
    """
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(port=Config.SERVER_PORT, debug=True)
