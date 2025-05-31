import os
import re
import json
import torch
import tempfile
import zipfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()


# ================================
# 1. Функции для работы с исходными файлами
# ================================
def find_source_files(root_dir, exts=('.c', '.cpp', '.h')):
    source_files = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(exts):
                source_files.append(os.path.join(subdir, file))
    return source_files

def concat_source_files(source_files):
    unified_lines = []
    file_line_offsets = {}
    for filepath in source_files:
        start_line = len(unified_lines) + 1
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            unified_lines.extend(lines)
        end_line = len(unified_lines)
        file_line_offsets[filepath] = (start_line, end_line)
    return unified_lines, file_line_offsets


# ================================
# 2. Функции предобработки текста
# ================================
def remove_comments(text):
    text_no_multiline = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return re.sub(r'//.*', '', text_no_multiline)

def remove_patterns(text):
    patterns = [
        r'(?i)(helperGood(?:G2B|B2G)?\d*|good(?:G2B|B2G)?\d*VaSink[BG]?)',
        r'(?i)(helperBad|badVaSink[BG]?)',
        r'(?i)(badSource(?:_[a-z])?|badSink(?:_[a-z])?|good(?:G2B\d*|B2G\d*)?Source(?:_[a-z])?|good(?:G2B\d*|B2G\d*)?Sink(?:_[a-z])?)',
        r'(?i)(BadClass|BadBaseClass|BadDerivedClass|GoodClass|GoodBaseClass|GoodDerivedClass)',
        r'(?i)CWE\d{0,3}',
        r'(?i)goodG2B\d*',
        r'(?i)goodB2G\d*',
        r'(?i)G2B\d{0,3}',
        r'(?i)B2G\d{0,3}',
        r'(?i)(good\d*|bad\d*)'
    ]
    for pat in patterns:
        text = re.sub(pat, '', text)
    return remove_comments(text)

def process_text(text):
    return remove_patterns(text)


# ================================
# 3. Маппинг предсказаний в CWE-метки
# ================================
def map_prediction_multi(preds):
    try:
        with open('class2id.json', 'r', encoding='utf-8') as f:
            class2id = json.load(f)
        id2class = {v: k for k, v in class2id.items()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки class2id.json: {str(e)}")

    predicted_labels = [id2class.get(idx, f"Unknown label {idx}") for idx, val in enumerate(preds) if val == 1]
    return predicted_labels or ["No vulnerability detected"]


# ================================
# 4. Инференс
# ================================
def run_inference_with_model(source_dir, tokenizer, model, device):
    source_files = find_source_files(source_dir)
    if not source_files:
        raise ValueError("Не найдено ни одного исходного файла (.c, .cpp, .h)")
    unified_lines, _ = concat_source_files(source_files)
    raw_text = "".join(unified_lines)
    processed_text = process_text(raw_text)

    tokenized = tokenizer(
        processed_text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_overflowing_tokens=True,
        stride=128
    )
    input_ids = torch.tensor(tokenized['input_ids']).to(device)
    attention_mask = torch.tensor(tokenized['attention_mask']).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probs = torch.sigmoid(logits).cpu().numpy()
    aggregated_probs = np.max(probs, axis=0)
    preds = (aggregated_probs > 0.5).astype(int)
    return map_prediction_multi(preds)


# ================================
# 5. Загрузка модели и токенизатора
# ================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained("miglss/codeBERT_CWE")
model.to(DEVICE)
model.eval()


# ================================
# 6. API endpoint
# ================================
@app.post("/classify")
async def classify_project(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Ожидается zip-архив с исходными файлами")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, file.filename)
            with open(zip_path, "wb") as f:
                f.write(await file.read())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            predicted_labels = run_inference_with_model(temp_dir, tokenizer, model, DEVICE)
            return {"vulnerabilities": predicted_labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok"}
