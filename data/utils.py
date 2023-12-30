from typing import List, Type
import pandas as pd
from info import (
    FileFormat,
    DataPath,
    DataName,
    TrainType,
    STSDatasetFeatures,
    TCDatasetFeatures,
    DatasetFeatures,
    UnsupervisedSimCseFeatures,
)
import os
import re


def get_folder_path(root: Type[DataPath], sub: Type[DataPath]) -> str:
    folder_path = os.path.join(root.value, sub.value)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    return folder_path


def get_data_path(
    folder_path: str,
    data_source: Type[DataName],
    train_type: Type[TrainType],
    file_format: Type[FileFormat],
) -> str:
    data_name = data_source.value + train_type.value + file_format.value
    data_path = os.path.join(folder_path, data_name)
    return data_path

#줄데이터 받아서 데이터프레임생성
def raw_data_to_dataframe(
    root_dir_name: Type[DataPath],
    raw_dir_name: Type[DataPath],
    data_source: Type[DataName],
    train_type: Type[TrainType],
    file_format: Type[FileFormat],
) -> pd.DataFrame:
    """
    make dataframe form raw data(klue-sts, KorSTS)
    [TODO]
    폴더에 해당 데이터가 있는지 검사 먼저
    """
    raw_floder_path = get_folder_path(root_dir_name, raw_dir_name)
    data_path = get_data_path(
        folder_path=raw_floder_path,
        data_source=data_source,
        train_type=train_type,
        file_format=file_format,
    )

    if file_format == FileFormat.JSON:
        rtn_data_frame = pd.read_json(data_path)
    elif file_format == FileFormat.TSV:
        rtn_data_frame = pd.read_csv(
            data_path, delimiter="\t", on_bad_lines="skip")

    return rtn_data_frame


def make_unsupervised_sentence_data(sts_data_list: List[pd.DataFrame]) -> List[str]:
    """
    sentence1, sentence2 의 column들을 한개로 통합
    """

    sentences = []
    for sts_data in sts_data_list:
        sentences.extend(
            sts_data[STSDatasetFeatures.SENTENCE1.value].to_list())
        sentences.extend(
            sts_data[STSDatasetFeatures.SENTENCE2.value].to_list())
    return sentences


def add_sts_df(sts_data_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    sentence1, sentence2 의 column들을 한개로 통합
    """

    sentence1 = []
    sentence2 = []
    for sts_data in sts_data_list:
        sentence1.extend(
            sts_data[STSDatasetFeatures.SENTENCE1.value].to_list())
        sentence2.extend(
            sts_data[STSDatasetFeatures.SENTENCE2.value].to_list())

    total_sts_df = pd.DataFrame(
        data={
            STSDatasetFeatures.SENTENCE1.value: sentence1,
            STSDatasetFeatures.SENTENCE2.value: sentence2,
        }
    )
    return total_sts_df


def change_col_name(
    df: pd.DataFrame,
    from_col_name: Type[DatasetFeatures],
    to_col_name: Type[DatasetFeatures],
):
    from_col_name = from_col_name.value
    to_col_name = to_col_name.value

    df[to_col_name] = df[from_col_name]
    del df[from_col_name]
    return df


# def save_preprocess_json_data(
#     sentences: List[pd.DataFrame], train_type: Type[TrainType]
# ) -> None:

#     root_dir_name = DataPath.ROOT.value
#     preprocess_dir_name = DataPath.PREPROCESS.value
#     preprocess_data_path = os.path.join(root_dir_name, preprocess_dir_name)
#     file_name = train_type.value + FileFormat.CSV.value
#     data_path = os.path.join(preprocess_data_path, file_name)

#     pd.DataFrame(data={UnsupervisedSimCseFeatures.SENTENCE.value: sentences}).to_csv(
#         data_path
#     )


#def wiki_preprocess(string):
#    """
#    한글, 한자, 많이 등장한 특수문자, 숫자를 제외하고 제거
 #   """
 #   s = re.sub(
 #       "[^0-9가-힣a-zA-Z一-龥.,)(\"'-·:}{》《~/%\[\]’‘“”=〉〈><_ ]", "", string)
 #   return s

def job_preprocess(string) :
    j = re.sub(
        "[^0-9가-힣a-zA-Z一-龥.,)(\"'-·:}{》《~/%\[\]’‘“”=〉〈><_ ]", "", string)
    return j




