import os
import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from konlpy.tag import Okt
from transformers import AutoTokenizer, AutoModel

# 데이터 로딩
df = pd.read_csv("tving_entertainment_all_merged.csv")
df["desc"] = df["description"].fillna("")
df["subgenre"] = df["subgenre"].fillna("")
df["genre"] = df["genre"].fillna("정보 없음")
df["cast"] = df["cast"].fillna("")

# 모델 로딩
okt = Okt()
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
model = AutoModel.from_pretrained("monologg/kobert", trust_remote_code=True)

# KoBERT 임베딩 함수
def get_kobert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# 형태소 필터링 함수
stopwords = {"이", "가", "은", "는", "을", "를", "의", "에", "에서", "으로", "입니다"}
def extract_nouns_filtered(text):
    nouns = okt.nouns(str(text))
    return ' '.join([w for w in nouns if w not in stopwords and len(w) > 1])

# KoBERT 임베딩
desc_embeddings = []
for desc in df["desc"]:
    try:
        vec = get_kobert_embedding(str(desc))
    except:
        vec = np.zeros(768)
    desc_embeddings.append(vec)
embedding_matrix = np.vstack(desc_embeddings)

# features_nouns 전처리
df["features"] = df[["desc", "genre", "subgenre"]].apply(
    lambda row: ' '.join([str(x) for x in row if x != "정보 없음"]), axis=1
)
df["features_nouns"] = df["features"].apply(extract_nouns_filtered)

# TF-IDF Vectorizer & Matrix
tfidf = TfidfVectorizer(min_df=3, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df["features_nouns"])

# 추천 함수
def hybrid_recommend(title, top_n=10, alpha=0.6, beta=0.3, gamma=0.1):
    if alpha + beta + gamma > 1.0:
        raise ValueError(f"alpha + beta + gamma는 1.0을 넘을 수 없습니다. 현재 합계: {alpha + beta + gamma}")

    if title not in df["title"].values:
        raise ValueError(f"{title} 이(가) 데이터에 없음")

    selected_idx = df[df["title"] == title].index[0]
    base_subgenre = df.iloc[selected_idx]["subgenre"]

    # KoBERT 유사도
    sim_kobert = cosine_similarity([embedding_matrix[selected_idx]], embedding_matrix)[0]

    # TF-IDF 유사도
    sim_tfidf = cosine_similarity(tfidf_matrix[selected_idx], tfidf_matrix).flatten()

    # CTR 가중치
    ctr_scores = np.ones(len(df))
    ctr_scaled = MinMaxScaler().fit_transform(ctr_scores.reshape(-1, 1)).flatten()

    # Subgenre Boost
    genre_boost = df["subgenre"].apply(lambda x: 1 if base_subgenre == "정보 없음" else (1.05 if x == base_subgenre else 1)).values

    # 최종 점수 계산
    boost_weight = 1.0 - alpha - beta - gamma
    final_scores = (
        sim_tfidf * alpha +
        sim_kobert * beta +
        ctr_scaled * gamma +
        (genre_boost - 1) * boost_weight
    )

    # 추천 결과 정리
    top_indices = np.argsort(final_scores)[::-1]
    results = []
    seen_titles = set()

    base_subgenres = set(df.iloc[selected_idx]["subgenre"].split(', ')) if df.iloc[selected_idx]["subgenre"] != "" else set()
    base_cast = set(df.iloc[selected_idx]["cast"].split(', ')) if df.iloc[selected_idx]["cast"] != "" else set()
    base_desc = set(df.iloc[selected_idx]["features_nouns"].split())

    for i in top_indices:
        vod_title = df.iloc[i]["title"]
        if vod_title == title or vod_title in seen_titles:
            continue
        seen_titles.add(vod_title)

        # 추천 근거 추출
        target_subgenres = set(df.iloc[i]["subgenre"].split(', ')) if df.iloc[i]["subgenre"] != "" else set()
        target_cast = set(df.iloc[i]["cast"].split(', ')) if df.iloc[i]["cast"] != "" else set()
        target_desc = set(df.iloc[i]["features_nouns"].split())

        genre_overlap = base_subgenres & target_subgenres
        cast_overlap = base_cast & target_cast
        desc_overlap = base_desc & target_desc

        reason = ""
        if genre_overlap:
            reason += f"장르 겹침: {list(genre_overlap)} "
        if cast_overlap:
            reason += f"출연진 겹침: {list(cast_overlap)} "
        if desc_overlap:
            reason += f"설명 키워드 겹침: {list(desc_overlap)[:3]}"

        results.append({
            "선택_프로그램": title,
            "추천_VOD": vod_title,
            "장르": df.iloc[i]["genre"],
            "서브장르": df.iloc[i]["subgenre"],
            "추천_근거": reason if reason else "유사도 기반 추천",
            "최종_점수": final_scores[i]
        })

        if len(results) == top_n:
            break

    return pd.DataFrame(results)

# 실행
result_df = hybrid_recommend("이혼숙려캠프", top_n=10, alpha=0.6, beta=0.3, gamma=0.1)
print(result_df)
