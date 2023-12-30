from enum import Enum


class DataName(Enum):
    """
    데이터를 얻은 곳: KLUE, KAKAO
    수정필요
    KLUE = klue
    KAKAO = kakao

    하지면 raw 데이터의 이름을 어떻게 나타낼 것인가? -> download_data.sh에서 이름이 바뀌어서 들어가게...
    """

    #PREPROCESS_WIKI = "wiki_"
    PREPROCESS_STS = "sts_"
    PREPROCESS_TC = "tc_"
    PREPROCESS_ADD = "add_"
    PREPROCESS_JOB = "job_"
    PREPROCESS_MYSELF = "myself_"
    #RAW_KLUE = "klue-sts-v1.1_"
    #RAW_KAKAO = "sts-"
    #RAW_TC = "ynat-v1.1_"
    RAW_COP = 'cop_'


class DataPath(Enum):
    """
    데이터 디렉토리 이름
    """

    ROOT = "data"
    RAW = "raw"
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class TrainType(Enum):
    """
    TRAIN:train
    DEV:dev
    TEST:test
    """

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class FileFormat(Enum):
    """
    JSON:.json
    TSV:.tsv
    """

    JSON = ".json"
    TSV = ".tsv"
    CSV = ".csv"


class DatasetFeatures(Enum):
    pass


class STSDatasetFeatures(DatasetFeatures):
    SENTENCE1 = "sentence1"
    SENTENCE2 = "sentence2"
    SCORE = "score"


class TCDatasetFeatures(DatasetFeatures):
    TITLE = "title"
    LABEL = "label"


class UnsupervisedSimCseFeatures(Enum):
    SENTENCE = "sentence"

class DataSource(Enum):
    JOB = "job"