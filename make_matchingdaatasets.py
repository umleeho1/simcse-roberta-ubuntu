import sys
import os
import pandas as pd
import torch

from data.info import UnsupervisedSimCseFeatures, STSDatasetFeatures
from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from SimCSE.arguments import DataTrainingArguments
from data.info import UnsupervisedSimCseFeatures, STSDatasetFeatures
from datasets import load_from_disk



# 감독학습 전처리
def sts_prepare_features(examples, tokenizer, data_args):
    total = len(examples[STSDatasetFeatures.SENTENCE1.value])
    scores = []
    for idx in range(total):
        score = examples[STSDatasetFeatures.SCORE.value][idx]
        if score is None:
            score = 0
        if examples[STSDatasetFeatures.SENTENCE1.value][idx] is None:
            examples[STSDatasetFeatures.SENTENCE1.value][idx] = " "
        if examples[STSDatasetFeatures.SENTENCE2.value][idx] is None:
            examples[STSDatasetFeatures.SENTENCE2.value][idx] = " "
        scores.append(score)

    tokenized_inputs = tokenizer(
        examples[STSDatasetFeatures.SENTENCE1.value],
        examples[STSDatasetFeatures.SENTENCE2.value],
        padding='max_length',
        max_length=data_args.max_seq_length,
        truncation=True,
        return_tensors='pt'
    )

    features = {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'token_type_ids': tokenized_inputs['token_type_ids'],
        'labels': scores
    }

    return features

def main(model_name_or_path, dev_file, save_dir):
    data_args = DataTrainingArguments(
        dev_file=dev_file,
        save_dir=save_dir,
        preprocessing_num_workers=4,
        overwrite_cache=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


    eval_data_files = {}

    if dev_file is not None:
        eval_data_files["dev"] = dev_file
        print("dev data success")
    else:
        print("no train data")
        return


    valid_extension = "csv"

    print("Train dataset loaded successfully.")
    print(f"Train file path: {dev_file}")

    eval_dataset = None

    if eval_data_files.get("dev") is not None:
        eval_dataset_path = "data/datasets/"
        eval_dataset = load_dataset(
            valid_extension,
            data_files=eval_data_files,
            cache_dir=None,
            delimiter=",",
        )["dev"]
        eval_dataset.save_to_disk(eval_dataset_path)


    dev_prepare_features_with_param = partial(
        sts_prepare_features, tokenizer=tokenizer, data_args=data_args
    )



    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            dev_prepare_features_with_param,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=["sentence1", "sentence2", "score"],
            load_from_cache_file=False,
        )
        eval_dataset.save_to_disk(data_args.save_dir + "/dev")

        # 추가: 평가 데이터셋의 샘플을 출력
        if eval_dataset is not None:
            print("Checking samples from the processed evaluation dataset:")
            for sample in eval_dataset.select(range(5)):
                print(sample)


if __name__ == "__main__":
    model_name_or_path = "kazma1/simcse-RobertLarge-job"
    dev_file = "dev.csv"  # 파일이 없으면 None으로 설정
    save_dir = "data/matchingdataset/good"

    main(model_name_or_path, dev_file,  save_dir)
    print("Dataset processing completed successfully.")
