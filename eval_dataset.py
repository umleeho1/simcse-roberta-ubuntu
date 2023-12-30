import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

def calculate_similarity(model, tokenizer, text1, text2):
    inputs = tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        similarity_score = torch.nn.functional.cosine_similarity(outputs.last_hidden_state[0], outputs.last_hidden_state[1], dim=1)
    return similarity_score

def create_eval_dataset(model_path, dataset_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    eval_examples = []

    dataset = pd.read_csv(dataset_path, delimiter="\t")

    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        text1 = row["sentence1"]
        text2 = row["sentence2"]
        score = float(row["score"])

        similarity_score = calculate_similarity(model, tokenizer, text1, text2)
        mean_similarity_score = round(similarity_score[0].item(), 2)  # 소수 둘째 자리 반올림
        # 소수 둘째 자리 반올림

        eval_examples.append({"sentence1": text1, "sentence2": text2, "score": mean_similarity_score, "true_score": score})

    return pd.DataFrame(eval_examples)

def main():
    model_path = "output/simcse-robertaLarge-matching/best_model"  # 여기에 모델 경로를 지정하세요.
    dataset_path = "data/dev/sts_dev.csv"  # 여기에 dev.csv 파일의 경로를 지정하세요.

    eval_dataset = create_eval_dataset(model_path, dataset_path)

    eval_dataset.to_csv("matchingresult-robertaLarge.csv", index=False)
    print(f"Eval dataset saved to matchingresult.csv")

if __name__ == "__main__":
    main()
