# IFITV_recommender

## 추천 파이프라인 흐름 (2025.04.14 기준)

### 📘 TF-IDF 기반 추천
1. 콘텐츠 메타데이터 통합 → title, description, genre, subgenre, cast
2. 추천 특징 조합 → description + genre + subgenre + cast → features
3. 형태소 분석 (Okt 기반) → 명사 추출 + 불용어 제거 → features_nouns
4. 텍스트 벡터화 → TfidfVectorizer(min_df=3, ngram_range=(1,2)) → tfidf_matrix
5. 콘텐츠 간 유사도 계산 → cosine_similarity(tfidf_matrix)
6. 추천 모델 → Weighted Genre Boosting 적용 (subgenre 겹침 수만큼 보정)
7. 추천 근거 생성 → 공통 subgenre + 출연진 + 설명 키워드 겹침 추출
8. 결과 출력 (DataFrame) → title, subgenre, 추천 근거

### 📙 Hybrid 추천 (TF-IDF + KoBERT)
9. KoBERT 문장 임베딩 생성 → description → kobert_matrix
10. KoBERT 유사도 계산 → cosine_similarity(kobert_matrix)
11. 하이브리드 유사도 계산 → α * tfidf_boosted + (1 - α) * kobert_similarity
12. Hybrid 추천 모델 → 추천 근거는 TF-IDF 기반 그대로 사용

---

## 주요 기능
- 티빙 예능 콘텐츠 메타데이터 전처리
- 형태소 분석기 Okt를 활용한 텍스트 정제
- TF-IDF + Cosine Similarity 기반 콘텐츠 유사도 계산
- 서브장르 Boosting, 추천 이유 생성 기능 포함
- KoBERT 기반 하이브리드 추천 지원

## 기술 스택
- Python 3.x
- pandas, scikit-learn
- konlpy (Okt)
- transformers (KoBERT)

## 사용 예시
```python
recommend("런닝맨", top_n=5, genre_weight=0.1)
hybrid_recommend_with_reason("나는 SOLO", top_n=5, alpha=0.7)
```

## 데이터 구성
- `title`: 프로그램명
- `description`: 설명
- `genre`: 상위 장르
- `subgenre`: 서브 장르
- `cast`: 주요 출연진
- `features_nouns`: 형태소 분석된 문자열
- 

---

필요하면 내가 바로 `.md` 파일 만들어서 줄 수도 있어.  
어떻게 해줄까? 😌
