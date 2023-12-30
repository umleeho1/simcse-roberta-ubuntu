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

# 비감독전처리함수
def unsupervised_prepare_features(examples, tokenizer, data_args):
    total = len(examples[UnsupervisedSimCseFeatures.SENTENCE.value])
    # Avoid "None" fields
    for idx in range(total):
        if examples[UnsupervisedSimCseFeatures.SENTENCE.value][idx] is None:
            examples[UnsupervisedSimCseFeatures.SENTENCE.value][idx] = " "

    sentences = (
        examples[UnsupervisedSimCseFeatures.SENTENCE.value]
        + examples[UnsupervisedSimCseFeatures.SENTENCE.value]
    )

    sent_features = tokenizer(
        sentences,
        max_length=data_args.max_seq_length,
        truncation=True,
        padding="max_length",
    )

    features = {}

    for key in sent_features:
        features[key] = [
            [sent_features[key][i], sent_features[key][i + total]] for i in range(total)
        ]
    return features

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

def main(model_name_or_path, train_file, dev_file, test_file, save_dir):
    data_args = DataTrainingArguments(
        train_file=train_file,
        dev_file=dev_file,
        test_file=test_file,
        save_dir=save_dir,
        preprocessing_num_workers=4,
        overwrite_cache=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    train_data_files = {}
    eval_data_files = {}
    if train_file is not None:
        train_data_files["train"] = train_file
        print("train data success")
    else:
        print("no train data")
        return

    if dev_file is not None:
        eval_data_files["dev"] = dev_file
        print("dev data success")
    else:
        print("no dev data")

    if test_file is not None:
        eval_data_files["test"] = test_file
        print("test data success")
    else:
        print("no test data")

    train_extension = train_file.split(".")[-1]
    valid_extension = "csv"

    train_dataset = load_dataset(
        train_extension,
        data_files=train_data_files,
        cache_dir=None,
    )

    print("Train dataset loaded successfully.")
    print(f"Train file path: {train_file}")

    eval_dataset = None

    if eval_data_files.get("dev") is not None:
        eval_dataset_path = "data/datasets/"
        eval_dataset = load_dataset(
            valid_extension,
            data_files=eval_data_files,
            cache_dir=None,
            delimiter="\t",
        )["dev"]
        eval_dataset.save_to_disk(eval_dataset_path)

    unsup_prepare_features_with_param = partial(
        unsupervised_prepare_features, tokenizer=tokenizer, data_args=data_args
    )
    dev_prepare_features_with_param = partial(
        sts_prepare_features, tokenizer=tokenizer, data_args=data_args
    )

    train_dataset = (
        train_dataset["train"]
        .map(
            unsup_prepare_features_with_param,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=UnsupervisedSimCseFeatures.SENTENCE.value,
            load_from_cache_file=False,
        )
        .save_to_disk(data_args.save_dir + "/train")
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
    model_name_or_path = "output/simcse-roberta/best_model"
    train_file = "data/train/job_train.csv"
    dev_file = "data/dev/sts_train.csv"  # 파일이 없으면 None으로 설정
    test_file = None  # 파일이 없으면 None으로 설정
    save_dir = "data/matchingdataset"

    main(model_name_or_path, train_file, dev_file, test_file, save_dir)
    print("Dataset processing completed successfully.")
