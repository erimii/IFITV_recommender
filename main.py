# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 12:36:03 2025

@author: Admin
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from konlpy.tag import Okt
okt = Okt()

from transformers import AutoTokenizer, AutoModel
import torch


# 한글 텍스트 → 명사만 추출 + 불용어 필터링 추가
stopwords = {"이", "가", "은", "는", "을", "를", "의", "에", "에서", "으로", "입니다"}

def extract_nouns_filtered(text):
    nouns = okt.nouns(str(text))
    return ' '.join([word for word in nouns if word not in stopwords and len(word) > 1])


# KoBERT 사전학습 모델
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
model = AutoModel.from_pretrained("monologg/kobert", trust_remote_code=True)

def get_kobert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # [CLS] 토큰의 임베딩 사용 (문장 전체 의미)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().numpy()



# CSV 로드
df = pd.read_csv("tving_entertainment_all_merged.csv")

# 결측값 채우기
df.fillna("정보 없음", inplace=True)

# 모든 콘텐츠 설명 벡터화
desc_embeddings = []
for desc in df['description']:
    try:
        vec = get_kobert_embedding(str(desc))
    except:
        vec = np.zeros(768)  # 오류 시 빈 벡터 처리
    desc_embeddings.append(vec)

embedding_matrix = np.vstack(desc_embeddings)

# 추천 특징 조합 (설명 + 장르+ 서브장르 + 출연진)
df["features"] = df[["description", "genre", "subgenre", "cast"]].apply(
    lambda row: ' '.join([str(x) for x in row if x != "정보 없음"]), axis=1)

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
cos_sim_kobert = cosine_similarity(embedding_matrix, embedding_matrix)

# title → index 매핑
title_to_index = pd.Series(df.index, index=df["title"])

# 추천 함수 정의
# TF-IDF만 사용
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
reslult = recommend("이혼숙려캠프", top_n=5)

# TF-IDF+ KoBERT 둘 다 사용
'''
1. TF-IDF 유사도 계산
2. Boosting (subgenre 겹침 가중치 추가)
3. KoBERT 유사도 계산
4. 두 유사도를 가중 평균 → final score
5. 추천 결과 + 이유 추출
'''
def hybrid_recommend_with_reason(title, top_n=5, alpha=0.7, genre_weight=0.1):
    idx = title_to_index[title]

    # 기준 정보
    base_subgenres = set(df.iloc[idx]["subgenre"].split(', ')) if df.iloc[idx]["subgenre"] != "정보 없음" else set()
    base_cast = set(df.iloc[idx]["cast"].split(', ')) if df.iloc[idx]["cast"] != "정보 없음" else set()
    base_desc = set(df.iloc[idx]["features_nouns"].split())

    boosted_tfidf = []
    reasons = []

    for i in range(len(df)):
        if i == idx:
            continue
        # 기본 TF-IDF 유사도
        tfidf_score = cos_sim[idx][i]
        kobert_score = cos_sim_kobert[idx][i]

        # Boosting 근거
        target_subgenres = set(df.iloc[i]["subgenre"].split(', ')) if df.iloc[i]["subgenre"] != "정보 없음" else set()
        target_cast = set(df.iloc[i]["cast"].split(', ')) if df.iloc[i]["cast"] != "정보 없음" else set()
        target_desc = set(df.iloc[i]["features_nouns"].split())

        genre_overlap = base_subgenres & target_subgenres
        cast_overlap = base_cast & target_cast
        desc_overlap = base_desc & target_desc

        # TF-IDF 보정 점수
        tfidf_boosted = tfidf_score + genre_weight * len(genre_overlap)

        # 하이브리드 최종 점수
        final_score = alpha * tfidf_boosted + (1 - alpha) * kobert_score
        boosted_tfidf.append((i, final_score))
        reasons.append((genre_overlap, cast_overlap, desc_overlap))

    # 정렬 및 결과 구성
    top_items = sorted(zip(boosted_tfidf, reasons), key=lambda x: x[0][1], reverse=True)[:top_n]

    results = []
    for ((i, score), (genres, casts, descs)) in top_items:
        result = {
            "title": df.iloc[i]["title"],
            "subgenre": df.iloc[i]["subgenre"],
            "추천 근거": ""
        }
        if genres:
            result["추천 근거"] += f"장르 겹침: {list(genres)} "
        if casts:
            result["추천 근거"] += f"출연진 겹침: {list(casts)} "
        if descs:
            result["추천 근거"] += f"설명 키워드 겹침: {list(descs)[:3]}"
        results.append(result)

    return pd.DataFrame(results)

# TF-IDF 70% + KoBERT 30% → TF-IDF 중심
reslult = hybrid_recommend_with_reason("이혼숙려캠프", top_n=5, alpha=0.7)


# 장르/출연진/설명 키워드 겹침 비율 비교
'''
나는 SOLO
이혼숙려캠프
뿅뿅 지구오락실 2
냉장고를 부탁해 since 2014
'''
def evaluate_reason_overlap(title, recommend_func, alpha=None):
    base = df[df["title"] == title].iloc[0]
    base_genre = set(base["subgenre"].split(', ')) if base["subgenre"] != "정보 없음" else set()
    base_cast = set(base["cast"].split(', ')) if base["cast"] != "정보 없음" else set()
    base_desc = set(base["features_nouns"].split())
    
    if alpha is None:
        result = recommend_func(title)
    else:
        result = recommend_func(title, alpha=alpha)

    overlaps = []
    for _, row in result.iterrows():
        target = df[df["title"] == row["title"]].iloc[0]
        g = set(target["subgenre"].split(', ')) if target["subgenre"] != "정보 없음" else set()
        c = set(target["cast"].split(', ')) if target["cast"] != "정보 없음" else set()
        d = set(target["features_nouns"].split())

        genre_overlap = len(base_genre & g)
        cast_overlap = len(base_cast & c)
        desc_overlap = len(base_desc & d)

        overlaps.append({
            "title": row["title"],
            "장르 겹침 수": genre_overlap,
            "출연진 겹침 수": cast_overlap,
            "설명 키워드 겹침 수": desc_overlap
        })

    return pd.DataFrame(overlaps)

print(evaluate_reason_overlap("냉장고를 부탁해 since 2014", recommend))
print(evaluate_reason_overlap("냉장고를 부탁해 since 2014", hybrid_recommend_with_reason, alpha=0.7))

