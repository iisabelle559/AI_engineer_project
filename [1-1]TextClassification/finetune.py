import json
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import numpy as np
from sklearn.metrics import classification_report
import os
import shutil

# 데이터셋 로드
with open("C:/Users/user/OneDrive/바탕 화면/TextClassification/data/230927_sentence_seg_dataset_1500.json", encoding='utf-8') as file:
    data = json.load(file)

# 라벨을 리스트로 변환하여 딕셔너리화
def label_list(label_name):
    return [label[0] for cf_label in data for label in cf_label['label'] if label[1] == label_name]

# 라벨 데이터 생성
label_names = ["a_med_rec", "a_eco_bg", "a_crim_rec", "prep_act", "a_mental_con", "motive", "crime_method", "reason_incmpl", "v_injury", "crime_result", "nan"]
data_dict = {label: label_list(label) for label in label_names}

# 컬럼 두개(label, value)로 데이터프레임화
df = pd.DataFrame([(key, value) for key, values in data_dict.items() for value in values], columns=['label_text', 'infor'])
df['infor'] = df['infor'].apply(lambda x: x if isinstance(x, list) else [x])
df = df.explode('infor').reset_index(drop=True)

# 라벨 숫자 변환
label2id = {label: idx for idx, label in enumerate(label_names)}
id2label = {idx: label for label, idx in label2id.items()}
df['labels'] = df['label_text'].map(label2id)

# 빠른 학습을 위해 데이터 샘플링 (데이터의 10%만 사용)
df_sample = df.sample(frac=0.1, random_state=42)

# Train/Test Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(df_sample, df_sample['labels']):
    train_df, test_eval_df = df_sample.iloc[train_index], df_sample.iloc[test_index]

# Test/Eval Split
ss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for test_index, eval_index in ss.split(test_eval_df, test_eval_df['labels']):
    test_df, eval_df = test_eval_df.iloc[test_index], test_eval_df.iloc[eval_index]

# 데이터셋 로드
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'test': Dataset.from_pandas(test_df),
    'val': Dataset.from_pandas(eval_df)
})

# '__index_level_0__' 열 제거
dataset = dataset.map(lambda x: {k: v for k, v in x.items() if k != '__index_level_0__'})

# 토크나이저
basemodel = 'klue/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(basemodel)

# 토큰화 함수
def preprocess_function(examples):
    return tokenizer(examples["infor"], truncation=True, padding=True)

# 데이터 토큰화
tokenized = dataset.map(preprocess_function, batched=True)
tokenized = tokenized.remove_columns(['label_text', 'infor'])

# 모델 로드 및 설정
model = AutoModelForSequenceClassification.from_pretrained(
    basemodel, num_labels=11, id2label=id2label, label2id=label2id
)

# 평가 메트릭 로드
f1_metric = evaluate.load("f1", average="micro")
precision_metric = evaluate.load("precision", average="micro")
recall_metric = evaluate.load("recall", average="micro")
accuracy_metric = evaluate.load("accuracy", average="micro")

# 메트릭 계산 함수
def compute_metrics(eval_pred):
    tmp_preds, labels = eval_pred
    preds = np.argmax(tmp_preds, axis=1)
    results = {
        "f1": f1_metric.compute(predictions=preds, references=labels, average='micro')["f1"],
        "precision": precision_metric.compute(predictions=preds, references=labels, average='micro')["precision"],
        "recall": recall_metric.compute(predictions=preds, references=labels, average='micro')["recall"],
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
    }
    print(classification_report(labels.tolist(), preds.tolist(), target_names=label_names))
    return results

# 파라미터 설정
training_args = TrainingArguments(
    output_dir='./result/checkpoints/',
    learning_rate=2e-5,
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,
    num_train_epochs=2,  
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["val"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# 모델 훈련
trainer.train()

# 가장 성능높은 모델 저장
best_model_path = trainer.state.best_model_checkpoint
if best_model_path is not None:
    shutil.copytree(best_model_path, './result/checkpoints/best_model', dirs_exist_ok=True)
    print(f"Best model saved to: ./result/checkpoints/best_model")
else:
    print("No best model checkpoint found.")
