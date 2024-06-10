from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import uvicorn
import threading
import time
import shutil
from torch.utils.data import DataLoader
from datasets import Dataset
import json
import evaluate
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# 모델 경로
model_path = './result/checkpoints/best_model' 
basemodel = 'klue/roberta-base'

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(basemodel)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
logging.info(f"Loaded model from {model_path}")

# 라벨 mapping
label_names = ["a_med_rec", "a_eco_bg", "a_crim_rec", "prep_act", "a_mental_con", "motive", "crime_method", "reason_incmpl", "v_injury", "crime_result", "nan"]
id2label = {idx: label for idx, label in enumerate(label_names)}
label2id = {label: idx for idx, label in enumerate(label_names)}

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 성능 평가를 위한 메트릭 로드
f1_metric = evaluate.load("f1")

# 테스트 데이터셋 로드
test_data_path = 'C:/Users/user/OneDrive/바탕 화면/TextClassification/data/230927_sentence_seg_dataset_1500.json' 
with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 테스트 데이터 전처리 및 토큰화
test_samples = []
for item in test_data:
    for text, label in item['label']:
        test_samples.append({'text': text, 'label': label2id[label]})

# 3% 샘플링
sample_size = int(0.03 * len(test_samples))
sampled_test_samples = random.sample(test_samples, sample_size)

sampled_texts = [sample['text'] for sample in sampled_test_samples]
sampled_labels = [sample['label'] for sample in sampled_test_samples]

def monitor_performance(interval=600):
    global model  
    while True:
        logging.info("Monitoring model performance...")

        if len(sampled_test_samples) < 10:
            logging.warning("Not enough data for performance monitoring.")
            time.sleep(interval)
            continue

        # 모델 성능 평가
        model.eval()
        predictions = []
        references = []
        test_encodings = tokenizer(sampled_texts, truncation=True, padding=True, return_tensors='pt')
        inputs = {k: v.to(model.device) for k, v in test_encodings.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        references = sampled_labels

        performance = f1_metric.compute(predictions=preds, references=references, average='micro')['f1']
        logging.info(f"Current model performance (F1): {performance}")

        threshold = 0.85  # 임계값 설정
        if performance < threshold:
            logging.info("Performance below threshold, deploying new model...")
            shutil.copytree('./result/checkpoints/checkpoint-180', './result/checkpoints/best_model', dirs_exist_ok=True)  # 새로운 모델 경로 설정
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
            logging.info("New model deployed.")

        time.sleep(interval)

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    logging.info(f"Received request for prediction: {text}")

    # 입력 텍스트 토큰화
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    # 모델 예측
    with torch.no_grad():
        outputs = model(**inputs)

    # 로짓을 소프트맥스 확률로 변환
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # 상위 3개 라벨의 인덱스 추출
    top_k_probs, top_k_indices = torch.topk(probs, 3, dim=-1)

    # 상위 3개 라벨과 해당 확률을 반환
    top_k_labels = [id2label[idx.item()] for idx in top_k_indices[0]]
    top_k_scores = top_k_probs[0].tolist()

    predictions = list(zip(top_k_labels, top_k_scores))
    logging.info(f"Prediction complete for input: {text}")

    # HTML 템플릿 렌더링
    return templates.TemplateResponse("result.html", {
        "request": request,
        "input_text": text,
        "predictions": predictions
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
