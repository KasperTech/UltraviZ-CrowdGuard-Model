from flask import Blueprint, jsonify,request,send_file
import os

def start_camera():
    # Logic to start the camera
    data = request.get_json()
    print(data)
    return jsonify({"status": "Camera started"}), 200


def stop_camera():
    # Logic to stop the camera
    data = request.get_json()
    return jsonify({"status": "Camera stopped"}), 200


def get_heatmap(camera_id):
    # Logic to get heatmap data
    temp_folder = 'temp'
    if not os.path.exists(temp_folder):
        return jsonify({"error": "Temp folder does not exist"}), 500
    else:
        return send_file(os.path.join(temp_folder, f"{camera_id}_heatmap.jpg"), mimetype='image/jpeg')
