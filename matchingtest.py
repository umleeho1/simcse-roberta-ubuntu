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

def get_highest_similarity_score(model, tokenizer, text1, text2_list):
    highest_score = -1.0  # 가장 높은 스코어를 저장할 변수 초기화
    for text2 in text2_list:
        similarity_score = calculate_similarity(model, tokenizer, text1, text2)
        similarity_score = similarity_score[0].item()
        if similarity_score > highest_score:
            highest_score = similarity_score
    return highest_score

def create_eval_dataset(model_path, dataset_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    eval_examples = []

    dataset = pd.read_csv(dataset_path, delimiter="\t")

    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        text1 = row["sentence1"]
        text2_list = dataset["sentence2"].tolist()  # 모든 sentence2 문장 리스트로 변환
        score = float(row["score"])

        highest_similarity_score = get_highest_similarity_score(model, tokenizer, text1, text2_list)
        highest_similarity_score = round(highest_similarity_score, 2)  # 소수 둘째 자리 반올림

        eval_examples.append({"sentence1": text1, "highest_score": highest_similarity_score, "true_score": score})

    return pd.DataFrame(eval_examples)

def main():
    model_path = "kazma1/simcse-robertsmall-matching"  # 여기에 모델 경로를 지정하세요.
    dataset_path = "data/dev/sts_dev.csv"  # 여기에 dev.csv 파일의 경로를 지정하세요.

    eval_dataset = create_eval_dataset(model_path, dataset_path)

    eval_dataset.to_csv("matchingresult.csv", index=False)
    print(f"Eval dataset saved to matchingresult.csv")

    # 평균 매칭률을 계산하여 출력
    avg_matching_score = eval_dataset["highest_score"].mean()
    print(f"Average Matching Score: {avg_matching_score:.2f}")

if __name__ == "__main__":
    main()