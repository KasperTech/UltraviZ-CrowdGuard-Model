from flask import Flask, jsonify, request
from routes.camera_routes import camera_bp

# Create the Flask app
app = Flask(__name__)
app.register_blueprint(camera_bp, url_prefix="/api/camera")


# Home route
@app.route('/')
def home():
    return "Hello, Flask server is running!"



# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5000)