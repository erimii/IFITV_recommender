# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 12:36:03 2025

@author: Admin
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
okt = Okt()

# 한글 텍스트 → 명사만 추출 + 불용어 필터링 추가
stopwords = {"이", "가", "은", "는", "을", "를", "의", "에", "에서", "으로", "입니다"}

def extract_nouns_filtered(text):
    nouns = okt.nouns(str(text))
    return ' '.join([word for word in nouns if word not in stopwords and len(word) > 1])


# CSV 로드
df = pd.read_csv("tving_entertainment_all_merged.csv")

df.info()

# 결측값 채우기
df.fillna("정보 없음", inplace=True)

# 추천 특징 조합 (설명 + 장르+ 서브장르 + 출연진)
df["features"] = df[["description", "genre", "subgenre", "cast"]].apply(
    lambda row: ' '.join([str(x) for x in row if x != "정보 없음"]), axis=1
)

# 형태소 분석 적용
df["features_nouns"] = df["features"].apply(extract_nouns_filtered)

# TF-IDF 벡터 생성
'''
min_df=3: 3개 미만 문서에 등장한 단어는 제거
ngram_range=(1,2): 유니그램 + 바이그램까지 반영
'''
tfidf = TfidfVectorizer(min_df=3, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df["features_nouns"])

# 코사인 유사도 계산
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# title → index 매핑
title_to_index = pd.Series(df.index, index=df["title"])

# 추천 함수 정의
def recommend(title, top_n=5, genre_weight=0.1):

    idx = title_to_index[title]
    sim_scores = list(enumerate(cos_sim[idx]))

    # 기준 예능의 서브장르 집합
    base_subgenres = set(df.iloc[idx]["subgenre"].split(', ')) if df.iloc[idx]["subgenre"] != "정보 없음" else set()

    boosted_scores = []
    for i, score in sim_scores:
        if i == idx:
            continue
        target_subgenres = set(df.iloc[i]["subgenre"].split(', ')) if df.iloc[i]["subgenre"] != "정보 없음" else set()
        overlap = base_subgenres & target_subgenres
        bonus = genre_weight * len(overlap)
        boosted_scores.append((i, score + bonus))

    boosted_scores = sorted(boosted_scores, key=lambda x: x[1], reverse=True)[:top_n]
    return df.iloc[[i[0] for i in boosted_scores]][["title", "subgenre"]]

print(recommend("이혼숙려캠프", top_n=5))




















