from flask import Blueprint, jsonify,request,send_file,Response
import os
from controller.model_initialise import camera_loop,generate_frames,stop_camera_fun,generate_heatmap_stream
import threading





def start_camera():
    # Logic to start the camera
    data = request.get_json()
    # print(data['camera_id'], data['stream_url'])
    thread = threading.Thread(
        target=camera_loop,
        args=(data['camera_id'],data['stream_url'], data['l1'], data['l2'],data['threshold'],data['camera_name']),
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
    return Response(generate_heatmap_stream(camera_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def get_video_feed(cam_id):
    return Response(generate_frames(cam_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
