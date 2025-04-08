from flask import Flask, request, jsonify, url_for
import os
import cv2
import numpy as np
from datetime import datetime
import torch
import concurrent.futures


app = Flask(__name__)

# 이미지 저장 디렉토리 설정
PROCESSED_DIR = os.path.join(app.root_path, 'static', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 두 모델 로드: 커스텀 모델과 기본 COCO 모델
custom_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/runs/train/exp11/weights/best.pt')
coco_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# custom_model.conf = 0.2

def file_to_image(file_storage):
    """업로드된 파일을 OpenCV 이미지로 변환"""
    file_data = file_storage.read()
    np_arr = np.frombuffer(file_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def save_image_to_file(img, prefix="processed"):
    """처리된 이미지를 저장하고 URL 반환"""
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    file_path = os.path.join(PROCESSED_DIR, filename)
    cv2.imwrite(file_path, img)
    file_url = url_for('static', filename=f"processed/{filename}", _external=True)
    return file_path, file_url

def detect_objects(img, target_object):
    """타겟 객체가 'strawberry'이면 커스텀 모델, 아니면 COCO 모델 사용하여 객체 감지"""
    if target_object.lower() == 'strawberry':
        selected_model = custom_model
    else:
        selected_model = coco_model
    results = selected_model(img)
    df = results.pandas().xyxy[0]  # Pandas DataFrame으로 결과 받기
    detections = df.to_dict(orient="records")
    return detections

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "Missing image file"}), 400
    img = file_to_image(request.files['image'])

    target_object = request.form.get('object')
    if not target_object:
        return jsonify({"error": "Missing object parameter"}), 400

    detections = detect_objects(img, target_object)
    # filtered = [d for d in detections if target_object.lower() in d['name'].lower()]
    # ****: 타겟이 strawberry인 경우, 모든 결과의 클래스 값을 0으로 변경
    if target_object.lower() == 'strawberry':
        for det in detections:
            det['class'] = 0  # 클래스 값을 0으로 설정
            det['name'] = 'strawberry'  # 이름도 strawberry로 확실하게 지정
    filtered = [d for d in detections if target_object.lower() in d['name'].lower()]

    _, file_url = save_image_to_file(img, prefix="detect")
    return jsonify({
        "fileUrl": file_url,
        "detections": filtered
    })

@app.route('/highlight', methods=['POST'])
def highlight():
    if 'image' not in request.files:
        return jsonify({"error": "Missing image file"}), 400
    img = file_to_image(request.files['image'])

    target_object = request.form.get('object')
    highlight_method = request.form.get('highlightMethod')
    if not target_object or not highlight_method:
        return jsonify({"error": "Missing parameters"}), 400

    detections = detect_objects(img, target_object)
    # ****: 타겟이 strawberry인 경우, 모든 결과의 클래스 값을 0으로 변경
    if target_object.lower() == 'strawberry':
        for det in detections:
            det['class'] = 0
            det['name'] = 'strawberry'
    filtered = [d for d in detections if target_object.lower() in d['name'].lower()]

    if highlight_method == "파란 테두리":
        for det in filtered:
            x1, y1 = int(det['xmin']), int(det['ymin'])
            x2, y2 = int(det['xmax']), int(det['ymax'])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 4)

    _, file_url = save_image_to_file(img, prefix="highlight")
    return jsonify({"fileUrl": file_url})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

from asgiref.wsgi import WsgiToAsgi
asgi_app = WsgiToAsgi(app)


# 실행 명령어
# python -m uvicorn main:asgi_app --reload