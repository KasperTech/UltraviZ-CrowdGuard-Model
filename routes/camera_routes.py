from flask import Blueprint, jsonify

from controller.camera_controller import start_camera,stop_camera,get_heatmap,get_video_feed

camera_bp = Blueprint("camera", __name__)

camera_bp.route("/start", methods=["POST"])(start_camera)
camera_bp.route('/stop', methods=["POST"])(stop_camera)
camera_bp.route('/heatmap/<string:camera_id>', methods=["GET"])(get_heatmap)
camera_bp.route('/video_feed/<cam_id>', methods=["GET"])(get_video_feed)
