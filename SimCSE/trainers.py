import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)
from datasets import load_metric
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers.utils import (
    logging,
)
from transformers.trainer_utils import EvalLoopOutput, has_length

from transformers import Trainer,EvalPrediction
from transformers.trainer_pt_utils import find_batch_size
import torch.distributed as dist

logger = logging.get_logger(__name__)



class CLTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:
        # 분산 환경이 초기화되었는지 확인
        is_distributed = torch.distributed.is_initialized()
        # eval_dataset = eval_dataset if not eval_dataset is None else self.eval_dataset
        self._memory_tracker.start()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        # 진단 코드: 첫 번째 배치의 데이터 구조 확인
        first_batch = next(iter(eval_dataloader))
        for k, v in first_batch.items():
            print(f"Batch key: {k}, shape: {v.shape}")

        # 진단 코드: 첫 번째 배치에 대한 모델 출력 확인
        with torch.no_grad():
            model_outputs = self.model(**{k: v.to(self.args.device) for k, v in first_batch.items()})
            print("Model outputs for the first batch:", model_outputs)

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # print("upper")
        sent1emb = []
        sent2emb = []
        scores = []
        model = self._wrap_model(self.model, training=False)
        model.eval()
        for i, data in enumerate(eval_dataloader):
            with torch.no_grad():
                for k in data:
                    label = data["labels"].view(data["labels"].size(0))
                    scores.extend(label.numpy())
                    # logger.info(data["input_ids"].shape)

                    input_ids1 = (
                        data["input_ids"][:, 0, :]
                        .view(-1, data["input_ids"].size(-1))
                        .to(self.args.device)
                    )
                    input_ids2 = (
                        data["input_ids"][:, 1, :]
                        .view(-1, data["input_ids"].size(-1))
                        .to(self.args.device)
                    )

                    attention_mask1 = (
                        data["attention_mask"][:, 0, :]
                        .view(-1, data["attention_mask"].size(-1))
                        .to(self.args.device)
                    )
                    attention_mask2 = (
                        data["attention_mask"][:, 1, :]
                        .view(-1, data["attention_mask"].size(-1))
                        .to(self.args.device)
                    )

                    if data["token_type_ids"] is not None:
                        token_type_ids1 = (
                            data["token_type_ids"][:, 0, :]
                            .view(-1, data["token_type_ids"].size(-1))
                            .to(self.args.device)
                        )
                        token_type_ids2 = (
                            data["token_type_ids"][:, 1, :]
                            .view(-1, data["token_type_ids"].size(-1))
                            .to(self.args.device)
                        )

                    output1 = model(
                        input_ids=input_ids1,
                        attention_mask=attention_mask1,
                        token_type_ids=token_type_ids1,
                        output_hidden_states=True,
                        return_dict=True,
                        sent_emb=True,
                    )
                    output2 = model(
                        input_ids=input_ids2,
                        attention_mask=attention_mask2,
                        token_type_ids=token_type_ids2,
                        output_hidden_states=True,
                        return_dict=True,
                        sent_emb=True,
                    )
                    pooler_output1 = output1.pooler_output.cpu()
                    pooler_output2 = output2.pooler_output.cpu()
                    sent1emb.append(pooler_output1)
                    sent2emb.append(pooler_output2)
        print(self.args.device)  #device확인
        print(type(self.args.device))
        sent1emb = torch.cat(sent1emb, 0).numpy()
        sent2emb = torch.cat(sent2emb, 0).numpy()

        cos_score = 1 - paired_cosine_distances(sent1emb, sent2emb)
        manhattan_distances = -paired_manhattan_distances(sent1emb, sent2emb)
        euclidean_distances = -paired_euclidean_distances(sent1emb, sent2emb)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(sent1emb, sent2emb)]

        pearsonr = load_metric("pearsonr").compute
        spearmanr = load_metric("spearmanr").compute

        all_scores = []
        all_scores.append(
            pearsonr(predictions=cos_score, references=scores)["pearsonr"]
        )
        all_scores.append(
            spearmanr(predictions=cos_score, references=scores)["spearmanr"]
        )
        all_scores.append(
            pearsonr(predictions=manhattan_distances, references=scores)["pearsonr"]
        )
        all_scores.append(
            spearmanr(predictions=manhattan_distances, references=scores)["spearmanr"]
        )

        all_scores.append(
            pearsonr(predictions=euclidean_distances, references=scores)["pearsonr"]
        )
        all_scores.append(
            spearmanr(predictions=euclidean_distances, references=scores)["spearmanr"]
        )

        all_scores.append(
            pearsonr(predictions=dot_products, references=scores)["pearsonr"]
        )
        all_scores.append(
            spearmanr(predictions=dot_products, references=scores)["spearmanr"]
        )

        all_scores = list(
            map(
                lambda x: torch.tensor(x, device=torch.device("cuda",self.args.local_rank)) if is_distributed else torch.tensor(x),
                all_scores,
            )
        )

        # 분산 환경에서만 실행되는 코드
        if is_distributed:
            mean_scores = []
            world_size = dist.get_world_size()  # 분산 환경에서 world_size 가져오기
            for score in all_scores:
                gather_t = [torch.ones_like(score) for _ in range(world_size)]
                dist.all_gather(gather_t, score)
                mean_scores.append(gather_t[0])  # 집계된 점수 사용
        else:
            mean_scores = all_scores  # 분산 환경이 아닐 경우, 원래 점수 사용
        print(mean_scores, self.args.local_rank)

        print(self.args.local_rank, "-", all_scores)

        metric_keys = [
            "eval_cosine_pearson",
            "eval_cosine_spearman",
            "eval_euclidean_pearson",
            "eval_euclidean_spearman",
            "eval_manhattan_pearson",
            "eval_manhattan_spearman",
            "eval_dot_pearson",
            "eval_dot_spearman",
        ]
        metric = {
            metric_key: mean_score.cpu().item()
            for metric_key, mean_score in zip(metric_keys, mean_scores)
        }
        output.metrics.update(metric)
        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics


