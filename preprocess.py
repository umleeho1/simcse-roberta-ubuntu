import torch
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from data.info import (
    DataName,
    DataPath,
    STSDatasetFeatures,
    TCDatasetFeatures,
    TrainType,
    FileFormat,
    UnsupervisedSimCseFeatures,
)
from data.utils import (
    get_data_path,
    get_folder_path,
    raw_data_to_dataframe,
    make_unsupervised_sentence_data,
    job_preprocess,
)
from transformers import AutoModel, AutoTokenizer

import logging

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)

# 채용공고_preprocess
df = pd.read_csv('cop.csv')
df['MergedColumn'] = df[['Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6']].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

job_dataset = df

data = []
for job_text in tqdm(job_dataset["MergedColumn"]):
    job_text = job_text.replace(". ", ".\n")
    job_text = job_text.replace("\xa0", " ")
    job_sentences = job_text.split("\n")

    for job_sentence in job_sentences:
        job_sentence = job_sentence.rstrip().lstrip()
        if len(job_sentence) >= 10:
            data.append(job_sentence)

job_df = pd.DataFrame(data={UnsupervisedSimCseFeatures.SENTENCE.value: data})
job_df[UnsupervisedSimCseFeatures.SENTENCE.value] = job_df[
    UnsupervisedSimCseFeatures.SENTENCE.value
].apply(job_preprocess)

train_floder_path = get_folder_path(root=DataPath.ROOT, sub=DataPath.TRAIN)
dev_floder_path = get_folder_path(root=DataPath.ROOT, sub=DataPath.DEV)
test_floder_path = get_folder_path(root=DataPath.ROOT, sub=DataPath.TEST)

preprocess_job_data_path = get_data_path(
    folder_path=train_floder_path,
    data_source=DataName.PREPROCESS_JOB,
    train_type=TrainType.TRAIN,
    file_format=FileFormat.CSV,
)

job_df = job_df.dropna(axis=0)
job_df.to_csv(preprocess_job_data_path, index=False)

logging.info(
    f"preprocess job train done!\nfeatures:{job_df.columns} \nlen: {len(job_df)}\nna count:{sum(job_df[UnsupervisedSimCseFeatures.SENTENCE.value].isna())}"
)

# 자기소개서
df = pd.read_csv('myself.csv')

data = []

for myself_text in tqdm(df['sentence']):
    myself_text = str(myself_text) if not pd.isna(myself_text) else ""
    myself_text = myself_text.replace(". ", ".\n")
    myself_text = myself_text.replace("\xa0", " ")
    myself_sentences = myself_text.split("\n")

    for myself_sentence in myself_sentences:
        myself_sentence = myself_sentence.strip()
        if len(myself_sentence) >= 10:
            data.append(myself_sentence)

myself_df = pd.DataFrame(data={UnsupervisedSimCseFeatures.SENTENCE.value: data})

myself_df[UnsupervisedSimCseFeatures.SENTENCE.value] = myself_df[
    UnsupervisedSimCseFeatures.SENTENCE.value
].apply(job_preprocess)

preprocess_myself_data_path = get_data_path(
    folder_path=train_floder_path,
    data_source=DataName.PREPROCESS_MYSELF,
    train_type=TrainType.TRAIN,
    file_format=FileFormat.CSV,
)

myself_df.to_csv(preprocess_myself_data_path, index=False)

logging.info(
    f"preprocess job train done! features:{myself_df.columns} \nlen: {len(myself_df)}\nna count:{sum(myself_df[UnsupervisedSimCseFeatures.SENTENCE.value].isna())}"
)

# sts_dev_df를 dev.csv 파일로부터 만들기
dev_df = pd.read_csv('dev.csv', encoding='utf-8')  # 파일 인코딩을 명시적으로 지정

# 'sentence1', 'sentence2' 열을 사용자 정의 전처리 함수로 처리
dev_df[STSDatasetFeatures.SENTENCE1.value] = dev_df['sentence1'].apply(job_preprocess)
dev_df[STSDatasetFeatures.SENTENCE2.value] = dev_df['sentence2'].apply(job_preprocess)

# 'label' 열을 그대로 사용
dev_df[STSDatasetFeatures.SCORE.value] = dev_df['score'].astype(float)

# 'label' 열을 제외하고 나머지 열을 포함하도록 선택
dev_df = dev_df[['sentence1', 'sentence2', 'score']].copy()

preprocess_sts_dev_path = get_data_path(
    dev_floder_path,
    data_source=DataName.PREPROCESS_STS,
    train_type=TrainType.DEV,
    file_format=FileFormat.CSV,
)

dev_df.to_csv(preprocess_sts_dev_path, sep="\t", index=False)

logging.info(
    f"preprocess sts dev done!\nfeatures:{dev_df.columns} \nlen: {len(dev_df)}"
)



# sts_dev_df를 dev.csv 파일로부터 만들기
dev_df = pd.read_csv('dev_train.csv', encoding='utf-8')  # 파일 인코딩을 명시적으로 지정

# 'sentence1', 'sentence2' 열을 사용자 정의 전처리 함수로 처리
dev_df[STSDatasetFeatures.SENTENCE1.value] = dev_df['sentence1'].apply(job_preprocess)
dev_df[STSDatasetFeatures.SENTENCE2.value] = dev_df['sentence2'].apply(job_preprocess)

# 'label' 열을 그대로 사용
dev_df[STSDatasetFeatures.SCORE.value] = dev_df['score'].astype(float)

# 'label' 열을 제외하고 나머지 열을 포함하도록 선택
dev_df = dev_df[['sentence1', 'sentence2', 'score']].copy()

preprocess_sts_dev_path = get_data_path(
    dev_floder_path,
    data_source=DataName.PREPROCESS_STS,
    train_type=TrainType.TRAIN,
    file_format=FileFormat.CSV,
)

dev_df.to_csv(preprocess_sts_dev_path, sep="\t", index=False)

logging.info(
    f"preprocess sts dev done!\nfeatures:{dev_df.columns} \nlen: {len(dev_df)}"
)
