import cv2
import torch

# 모델 로드 (예: 커스텀 모델)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/runs/train/exp5/weights/best.pt')

# 테스트 이미지 로드
img_path = '/Users/dgsw38/IdeaProjects/lookback-ai/fruits.jpg'
img = cv2.imread(img_path)

model.conf = 0.1

results = model(img)
df = results.pandas().xyxy[0]
print(df)