
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# TIBERO DB
import pyodbc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


db = pyodbc.connect('DSN=dajoba')

# 인코딩 설정
#db.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
#db.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
db.setdecoding(pyodbc.SQL_CHAR, encoding='cp949')
db.setdecoding(pyodbc.SQL_WCHAR, encoding='cp949')
db.setencoding(encoding='utf-8')
cursor = db.cursor()

cursor.execute("SELECT * FROM JOB_POSTING")
row = cursor.fetchone()

print(row)  # 행의 모든 데이터를 출력


# SIGNAL 값이 1인 레코드가 있을 때까지 반복
#필요한 자소서 정보 가져오기
# while True:
#     cursor.execute("SELECT USER_ID,INTRO_ID,DESIRE_FIELD,INTRO_CONTENT FROM SELF_INTRODUCTION WHERE SIGNAL = 1")
#     row = cursor.fetchone() # 첫 번째 결과 가져오기

#     if row is None:
#         print("complete all signal")
#         break # 더 이상 SIGNAL이 1인 레코드가 없으면 반복 중단
#     # 각 레코드의 값을 사용하여 필요한 작업 수행
#     user_id = row[0]   # 유저ID type:number
#     intro_id = row[1]  # 자기소개서ID type:number
#     desire_field = row[2] #희망분야  type:numer
#     intro_cuntent = row[3]#자기소개서 type:CLOB

#     print("자기소개서 정보") #레코드 간 구분을 위한 빈 줄
#     print(f"User ID: {user_id}")
#     print(f"Intro ID: {intro_id}")
#     print(f"Desire Field: {desire_field}")
#     print(f"Intro Content: {intro_cuntent[:200]}")  # 자기소개서 내용이 길 경우, 처음 200자만 출력



#     # SIGNAL이 1인 레코드를 사용하여 필요한 작업 수행
#     # JOB_POSTING 테이블에서 현재 SELF_INTRODUCTION 레코드의 DESIRE_FIELD 값과 일치하는 레코드 찾기
#     query = "SELECT TITLE, GROUP_INTRO, MAINDUTIES, QUALIFICATION, PREFERENTIAL FROM JOB_POSTING WHERE JOB_GROUP = ?"
#     cursor.execute(query, (desire_field,))
#     matching_job_postings = cursor.fetchall()

#     # 각 레코드를 순회하면서 컬럼 값 추출
#     for job_posting in matching_job_postings:
#         job_title = job_posting[0]
#         group_intro = job_posting[1]
#         mainduties = job_posting[2]
#         qualification = job_posting[3]
#         preferential = job_posting[4]


#         # 각 레코드의 정보를 출력 또는 처리
#         print("JOB_POSTING")  # 레코드 간 구분을 위한 빈 줄
#         print(
#             f"Job Title: {job_title}, Group Intro: {group_intro}, Main Duties: {mainduties}, Qualification: {qualification}, Preferential: {preferential}")


#     # 필요한 처리 수행...



#     # 처리가 완료된 후 SIGNAL 값을 0으로 업데이트
#     cursor.execute("UPDATE SELF_INTRODUCTION SET SIGNAL = 1 WHERE USER_ID = ? AND INTRO_ID = ?", (user_id, intro_id))
#     db.commit()  # 변경 사항을 데이터베이스에 반영

# # 반복이 끝나면 필요한 후처리 수행
# cursor.execute("SELECT * FROM JOB_POSTING AS J,SELF_INTRODUCTION AS S WHERE J.JOB_GROUP = S.DESIRE_FIELD")

