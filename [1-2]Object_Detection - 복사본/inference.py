from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import shutil
import logging
import uvicorn
import threading
import time
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# 추론을 진행할 모델 경로
best_model_path = "training_resultV2/nano_3_epochs/train/weights/best.pt"

# 모델 로드
model = YOLO(best_model_path)
logging.info(f"Loaded model from {best_model_path}")

# 템플릿 설정
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 테스트 데이터셋
test_data_dir = 'C:/Users/user/OneDrive/바탕 화면/Object_Detection/Aquarium Combined.v2-raw-1024.yolov8/test' 
test_images_dir = os.path.join(test_data_dir, 'images')
test_labels_dir = os.path.join(test_data_dir, 'labels')

test_images = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith('.jpg') or img.endswith('.png')]  # 테스트 이미지
test_labels = [os.path.join(test_labels_dir, lbl) for lbl in os.listdir(test_labels_dir) if lbl.endswith('.txt')]  # 테스트 라벨

# 이미지 파일과 라벨 파일이 일치하는지 확인
def get_matching_labels(image_files, label_files):
    image_base_names = {os.path.splitext(os.path.basename(img))[0]: img for img in image_files}
    label_base_names = {os.path.splitext(os.path.basename(lbl))[0]: lbl for lbl in label_files}
    
    matching_images = []
    matching_labels = []
    
    for base_name, img_path in image_base_names.items():
        if base_name in label_base_names:
            matching_images.append(img_path)
            matching_labels.append(label_base_names[base_name])
    
    return matching_images, matching_labels

test_images, test_labels = get_matching_labels(test_images, test_labels)

def monitor_performance(interval=600):
    global model
    while True:
        logging.info("Monitoring model performance...")

        if len(test_images) < 10 or len(test_labels) < 10:
            logging.warning("Not enough data for performance monitoring.")
            time.sleep(interval)
            continue

        if len(test_images) != len(test_labels):
            logging.warning("Mismatch between number of test images and labels.")
            time.sleep(interval)
            continue

        # 모델 성능 평가
        correct_predictions = 0
        total_predictions = len(test_images)

        for img_path, label_path in zip(test_images, test_labels):
            img = cv2.imread(img_path)
            results = model(img)

            # 첫 번째 결과만 사용
            result = results[0]
            
            # 실제 라벨
            with open(label_path, 'r') as f:
                true_labels = [line.strip().split()[0] for line in f.readlines()]
            
            # YOLO에서 예측된 라벨
            predicted_labels = [result.names[int(cls)] for cls in result.boxes.cls]

            # 위 두 라벨 일치 여부 확인
            correct_predictions += sum(1 for pl in predicted_labels if pl in true_labels)

        accuracy = correct_predictions / total_predictions
        logging.info(f"Current model performance (Accuracy): {accuracy}")

        if accuracy < 0.5:
            print("Performance below threshold, deploying new model...")
            
            try:
                # 임계치보다 낮으면 새로운 모델불러오기
                best_model_path = 'training_resultV2/nano_10_epochs/train/weights/best.pt'
                model = YOLO(best_model_path)
                print("New model deployed successfully.")
            except Exception as e:
                print(f"Error deploying new model: {e}")

        time.sleep(interval)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    logging.info(f"Received request for prediction")
    # 이미지 저장
    image_path = f"static/{file.filename}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 이미지 로드
    img = cv2.imread(image_path)

    # 추론 실행
    results = model(img)

    # 결과를 이미지에 시각화
    annotated_frame = results[0].plot()
    result_image_path = f"static/annotated_{file.filename}"
    cv2.imwrite(result_image_path, annotated_frame)

    logging.info(f"Prediction complete for {file.filename}")

    # HTML 템플릿 렌더링
    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image": f"/static/{file.filename}",
        "result_image": f"/static/annotated_{file.filename}"
    })

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", response_class=JSONResponse)
async def health_check():
    return JSONResponse(status_code=200, content={"status": "ok"})

if __name__ == "__main__":
    monitor_thread = threading.Thread(target=monitor_performance)
    monitor_thread.daemon = True
    monitor_thread.start()
    uvicorn.run(app, host='0.0.0.0', port=8000)

