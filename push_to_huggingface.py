from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# 모델과 토크나이저 로드
model = AutoModelForSequenceClassification.from_pretrained("output/simcse-robertasmall-matching/best_model")
tokenizer = AutoTokenizer.from_pretrained("output/simcse-robertasmall-matching/best_model")

# 모델 업로드
model.save_pretrained("kazma1/simcse-robertasmall-matching")
tokenizer.save_pretrained("kazma1/simcse-robertasmall-matching")
