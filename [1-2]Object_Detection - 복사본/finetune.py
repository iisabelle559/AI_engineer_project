from ultralytics import YOLO
import os
import torch

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
nano_model = YOLO("yolov8n.pt")  # nano
nano_model.to(device)

# 파라미터 설정
training_configs = {
    "nano": {"epochs": 3, "batch_size": 16, "learning_rate": 0.01, "optimizer": 'Adam', "verbose": True}
}

# 체크포인트 저장 경로
results_base_dir = "./training_resultV2"

# Train/Evaluate 함수
def train_and_evaluate(model, model_name, config):
    print(f"Training {model_name} model for {config['epochs']} epochs with batch size {config['batch_size']} and learning rate {config['learning_rate']}...")
    
    # Define a unique directory for each training session
    results_dir = os.path.join(results_base_dir, f"{model_name}_{config['epochs']}_epochs")
    os.makedirs(results_dir, exist_ok=True)
    
    model.train(
        data="C:/Users/user/OneDrive/바탕 화면/Object_Detection/Aquarium Combined.v2-raw-1024.yolov8/data.yaml",  
        epochs=config['epochs'],
        batch=config['batch_size'],
        project=results_dir,
        name="train",
        lr0=config['learning_rate'],
        optimizer=config['optimizer'],
        verbose=config['verbose']
    )
    
    # 평가
    print(f"Evaluating {model_name} model...")
    metrics = model.val(project=results_dir, name="val")
    
    # 결과 저장
    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write(str(metrics))

# 학습/평가
train_and_evaluate(nano_model, "nano", training_configs["nano"])

print("Training and evaluation complete. Results saved in respective directories.")
