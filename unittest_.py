from typing import List, Type
import pandas as pd
import data.info
import re

from data.utils import get_data_path, get_folder_path


class DataSource:
    pass


if __name__ == "__main__":
    # DataPath.ROOT,DataPath.RAW,DataSource.KLUE, TrainType.TRAIN, FileFormat.JSON
    raw_floder_path = get_folder_path(data.info.DataPath.ROOT, data.info.DataPath.RAW)
    data_path = get_data_path(
        folder_path=raw_floder_path,
        data_source=DataSource.KLUE,
        train_type=data.info.TrainType.TRAIN,
        file_format=data.info.FileFormat.JSON,
    )

    print(data_path)
    pass
