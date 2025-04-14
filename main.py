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
def recommend(title, top_n=5, genre_weight=0.1, show_reason=True):
    idx = title_to_index[title]
    sim_scores = list(enumerate(cos_sim[idx]))

    base_subgenres = set(df.iloc[idx]["subgenre"].split(', ')) if df.iloc[idx]["subgenre"] != "정보 없음" else set()
    base_cast = set(df.iloc[idx]["cast"].split(', ')) if df.iloc[idx]["cast"] != "정보 없음" else set()
    base_desc_keywords = set(df.iloc[idx]["features_nouns"].split())

    boosted_scores = []
    reasons = []

    for i, score in sim_scores:
        if i == idx:
            continue

        target_subgenres = set(df.iloc[i]["subgenre"].split(', ')) if df.iloc[i]["subgenre"] != "정보 없음" else set()
        target_cast = set(df.iloc[i]["cast"].split(', ')) if df.iloc[i]["cast"] != "정보 없음" else set()
        target_desc_keywords = set(df.iloc[i]["features_nouns"].split())

        genre_overlap = base_subgenres & target_subgenres
        cast_overlap = base_cast & target_cast
        desc_overlap = base_desc_keywords & target_desc_keywords

        bonus = genre_weight * len(genre_overlap)
        boosted_scores.append((i, score + bonus))
        reasons.append((genre_overlap, cast_overlap, desc_overlap))

    top_items = sorted(zip(boosted_scores, reasons), key=lambda x: x[0][1], reverse=True)[:top_n]

    results = []
    for ((i, score), (genres, casts, desc)) in top_items:
        result = {
            "title": df.iloc[i]["title"],
            "subgenre": df.iloc[i]["subgenre"],
        }
        if show_reason:
            reason_str = ""
            if genres:
                reason_str += f"장르 겹침: {list(genres)} "
            if casts:
                reason_str += f"출연진 겹침: {list(casts)} "
            if desc:
                reason_str += f"설명 키워드 겹침: {list(desc)[:3]}"  # 너무 길면 상위 3개만
            result["추천 근거"] = reason_str.strip()
        results.append(result)

    return pd.DataFrame(results)



print(recommend("이혼숙려캠프", top_n=5))




















