from info import UnsupervisedSimCseFeatures, STSDatasetFeatures

#비감독전처리함수
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

#감독학습 전처리
def sts_prepare_features(examples, tokenizer, data_args):
    total = len(examples[STSDatasetFeatures.SENTENCE1.value])
    # Avoid "None" fields
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

    sentences = (
        examples[STSDatasetFeatures.SENTENCE1.value]
        + examples[STSDatasetFeatures.SENTENCE2.value]
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
    features["labels"] = scores

    """
    ex)
    {'input_ids': [[1,2,3,4,5],[1,2,3,4,5]] 
    'token_type_ids': [[1,1,1,1,1],[1,1,1,1,1]], 
    'attention_mask': [[1,1,1,1,1],[1,1,1,1,1]], 
    'label': 1.0}
    """

    return features
