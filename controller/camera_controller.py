from flask import Blueprint, jsonify,request,send_file,Response
import os
from controller.model_initialise import camera_loop,generate_frames,stop_camera_fun
import threading





def start_camera():
    # Logic to start the camera
    data = request.get_json()
    # print(data['camera_id'], data['stream_url'])
    thread = threading.Thread(
        target=camera_loop, 
        args=(data['camera_id'],data['stream_url'], data['l1'], data['l2']), 
        daemon=True  # dies when main process exits
    )
    thread.start()
    # camera_loop(data['camera_id'], data['stream_url'], data['l1'], data['l2'])
    return jsonify({"status": "Camera started"}), 200


def stop_camera():
    # Logic to stop the camera
    data = request.get_json()
    print(data['camera_id'])
    stop_camera_fun(data['camera_id'])
    return jsonify({"status": "Camera stopped"}), 200


def get_heatmap(camera_id):
    # Logic to get heatmap data
    temp_folder = 'temp'
    if not os.path.exists(temp_folder):
        return jsonify({"error": "Temp folder does not exist"}), 500
    else:
        return send_file(os.path.join(temp_folder, f"{camera_id}_heatmap.jpg"), mimetype='image/jpeg')


def get_video_feed(cam_id):
    return Response(generate_frames(cam_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
