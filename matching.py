import pyodbc
from transformers import AutoModel, AutoTokenizer
import torch
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ... (calculate_similarity, get_highest_similarity_score 함수 정의)
def calculate_similarity(model, tokenizer, text1, text2):
    inputs = tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        similarity_scores = torch.nn.functional.cosine_similarity(outputs.last_hidden_state[0], outputs.last_hidden_state[1], dim=1)
        similarity_score = similarity_scores.mean()  # 여러 점수의 평균을 계산
    return similarity_score

def calculate_max_similarity_per_sentence(model, tokenizer, intro_text, job_posting_text):
    intro_sentences = sent_tokenize(intro_text)  
    job_posting_sentences = sent_tokenize(job_posting_text)

    max_scores = []
    for intro_sentence in intro_sentences:
        sentence_scores = [
            calculate_similarity(model, tokenizer, intro_sentence, job_posting_sentence).item()
            for job_posting_sentence in job_posting_sentences
        ]
        max_scores.append(max(sentence_scores))

    return sum(max_scores) / len(max_scores) if max_scores else 0

def calculate_highest_similarity_score(model, tokenizer, intro_text, job_posting_text):
    intro_sentences = sent_tokenize(intro_text)
    job_posting_sentences = sent_tokenize(job_posting_text)

    highest_score = 0
    for intro_sentence in intro_sentences:
        for job_posting_sentence in job_posting_sentences:
            current_score = calculate_similarity(model, tokenizer, intro_sentence, job_posting_sentence).item()
            logging.info(f"Comparing: '{intro_sentence}' with '{job_posting_sentence}' - Score: {current_score}")
            if current_score > highest_score:
                highest_score = current_score
                logging.info(f"New highest score found: {highest_score}")

    logging.info(f"Final highest score: {highest_score}")
    return highest_score



# db 연결
db = pyodbc.connect('DSN=dajoba')

# 인코딩 설정
# db.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
# db.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
db.setdecoding(pyodbc.SQL_CHAR, encoding='cp949')
db.setdecoding(pyodbc.SQL_WCHAR, encoding='cp949')
db.setencoding(encoding='utf-8')
cursor = db.cursor()

 # 트랜스포머 모델 및 토크나이저 초기화
model_path = "kazma1/simcse-robertsmall-matching"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# ...
while True:
    cursor.execute("SELECT USER_ID, INTRO_ID, DESIRE_FIELD, INTRO_CONTENT FROM SELF_INTRODUCTION WHERE SIGNAL = 1")
    rows = cursor.fetchall()
    
    print(f"Retrieved {len(rows)} rows with SIGNAL = 1.")

    if not rows:
        print("All signals processed.")
        continue

    for row in rows:
        user_id = row[0]
        intro_id = row[1]
        desire_field = row[2]
        intro_content = row[3]

        print(f"Processing intro_id: {intro_id}")

        # 해당 희망분야의 채용공고 불러오기
        cursor.execute(
            "SELECT JOB_POSTING_ID, TITLE, GROUP_INTRO, MAINDUTIES, QUALIFICATION, PREFERENTIAL FROM JOB_POSTING WHERE JOB_GROUP = ?",
            desire_field)
        job_postings = cursor.fetchmany(100)


        # 나머지 채용공고와의 매칭 점수 계산
        for posting in job_postings:
            job_posting_id = posting[0]
            job_posting_content = ' '.join(item if item is not None else '' for item in posting[1:])

            mainduties_scores = calculate_highest_similarity_score(model, tokenizer, intro_content, posting[3])
            
            if mainduties_scores >= 0.3:

                #max_similarity_score = calculate_max_similarity_per_sentence(model, tokenizer, intro_content,
                #                                                        job_posting_content)
                #cursor.execute("INSERT INTO MATCH (MATCH_SCORE, JOB_POSTING_ID, INTRO_ID) VALUES (?, ?, ?)",
                #           (max_similarity_score, job_posting_id, intro_id))
                cursor.execute("INSERT INTO MATCH (MATCH_SCORE, JOB_POSTING_ID, INTRO_ID) VALUES (?, ?, ?)",
                           (mainduties_scores, job_posting_id, intro_id))
                db.commit()
                continue
            else:
                #matching_scores = calculate_max_similarity_per_sentence(model, tokenizer, intro_content, posting[3])
                #max_similarity_score = 0
                #print(f"Job Posting ID: {job_posting_id}, Main Duties Score: {mainduties_scores}")
                #cursor.execute("INSERT INTO MATCH (MATCH_SCORE, JOB_POSTING_ID, INTRO_ID) VALUES (?, ?, ?)",
                #           (matching_scores, job_posting_id, intro_id))
                cursor.execute("INSERT INTO MATCH (MATCH_SCORE, JOB_POSTING_ID, INTRO_ID) VALUES (?, ?, ?)",
                           (mainduties_scores, job_posting_id, intro_id))
                db.commit()
                
                continue

           
            
            

        # SIGNAL 값을 0으로 업데이트 (자소서의 검사가 완료된 후에 업데이트)
        cursor.execute("UPDATE SELF_INTRODUCTION SET SIGNAL = 0 WHERE INTRO_ID = ?", intro_id)
        db.commit()
     

print("Matching results saved to the MATCH table")