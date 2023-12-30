
# ubuntu 
환경에서 분산학습환경 제공 odbc로 데어터관리  

# roberta-simcse
자기소개서+채용공고 도메인학습  

# simcse-roberta-matching
dev_train.csv 자기소개서  채용공고 스코어 형태의 유사도비교 데이터학습  

# 검증데이터 
dev.csv  자기소개서 채용공고 스코어 형태의 유사도 데이터  

# 검증결과
| Model                      | Cosine Pearson | Cosine Spearman | Euclidean Pearson | Euclidean Spearman | Manhattan Pearson | Manhattan Spearman | Dot Pearson | Dot Spearman |
|----------------------------|----------------|-----------------|-------------------|--------------------|-------------------|--------------------|-------------|--------------|
| SimCSE-RoBERTalarge-matching            | 61.23          | 54.12           | 53.37             | 53.32              | 63.14             | 61.82              | 52.14       | 51.33        |
| SimCSE-RoBERTasmall-matching  | 53.12          | 53.09           | 52.31             | 53.24              | 52.81             | 53.65              | 48.08       | 48.35        |  

# reference  

```python
@article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={arXiv preprint arXiv:2004.03289},
  year={2020}
}
```


```python
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation},
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook
              Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and
              Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin
              Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
