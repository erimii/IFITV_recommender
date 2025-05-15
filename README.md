
## IFITV Recommender - Hybrid 최종버전 (2025.05 기준)

# IFITV Recommender

**티빙 예능 콘텐츠 기반 맞춤형 추천 시스템**
→ TF-IDF + KoBERT Hybrid 모델 기반 최종 버전

---

## 추천 파이프라인

### 데이터 전처리

* **콘텐츠 메타데이터 로드**: title, description, genre, subgenre, cast
* 결측값 처리 → '정보 없음' / 빈 문자열로 채움
* **features 구성**: description + genre + subgenre 조합
* **형태소 분석 (Okt)**: 명사 추출 + 불용어 제거 → `features_nouns`
* **TF-IDF 벡터화**: min\_df=3, ngram\_range=(1,2)
* **KoBERT 임베딩 생성**: description 기반 embedding\_matrix

---

### Hybrid 추천 모델 (hybrid\_recommend)

* **TF-IDF 유사도 계산**: cosine\_similarity(tfidf\_matrix)

* **KoBERT 유사도 계산**: cosine\_similarity(embedding\_matrix)

* **CTR 가중치 적용**: (현재는 1로 고정, 확장성 대비 구조화)

* **Subgenre Boost 적용**: 기준 콘텐츠와 subgenre 일치 시 boost

* **최종 점수 계산**:

  ```
  final_score = (alpha * TF-IDF 유사도) +
                (beta * KoBERT 유사도) +
                (gamma * CTR) +
                (subgenre boost × boost_weight)
  ```

  * boost\_weight = 1 - alpha - beta - gamma

* **추천 근거 생성**:

  * 장르 겹침
  * 출연진 겹침
  * 설명 키워드 겹침

* **최종 결과 출력**: 추천 콘텐츠명, 장르, 서브장르, 추천 근거, 최종 점수

---

## 주요 기능 요약

| 기능                             | 설명                                             |
| ------------------------------ | ---------------------------------------------- |
| TF-IDF & KoBERT Hybrid 추천      | 형태소 기반 TF-IDF + 의미 기반 KoBERT 결합                |
| 가중치 튜닝 가능 (alpha, beta, gamma) | 추천 전략 유연 조정                                    |
| CTR 가중치 구조 포함                  | 추후 실사용 클릭률 반영 가능                               |
| Subgenre Boost                 | 장르 일치 시 추가 가중치                                 |
| 추천 근거 (장르/출연진/키워드) 표시          | 사용자가 추천 이유를 명확히 이해 가능                          |
| 캐싱 최적화                         | embedding\_matrix, tfidf\_matrix 사전 계산으로 성능 향상 |

---

## 사용 예시

```python
# Hybrid 추천 호출
result_df = hybrid_recommend("이혼숙려캠프", top_n=10, alpha=0.6, beta=0.3, gamma=0.1)
print(result_df)
```

---

## 데이터 컬럼 설명

| 컬럼명               | 설명                                |
| ----------------- | --------------------------------- |
| title             | 프로그램명                             |
| description       | 콘텐츠 설명                            |
| genre             | 상위 장르                             |
| subgenre          | 세부 장르                             |
| cast              | 주요 출연진                            |
| features          | description + genre + subgenre 조합 |
| features\_nouns   | Okt 형태소 분석 후 불용어 제거 결과            |
| embedding\_matrix | KoBERT 임베딩 결과                     |
| tfidf\_matrix     | TF-IDF 벡터화 결과                     |

---

## 변경 이력

* 2025.05: Hybrid 추천 최적화 (CTR 구조 도입, 가중치 로직 명확화, 캐싱 최적화)
* 추천 근거(장르/출연진/키워드 겹침)는 기존 방식 유지
* 최종 점수 및 가중치 구조 명시화

---

## 파일 구성

| 파일명                                        | 설명                       |
| ------------------------------------------ | ------------------------ |
| `main.py`                                  | 초기 TF-IDF+ KoBERT 기반 추천 코드       |
| `hybrid_final_tving_recommender_better.py` | Hybrid 추천 최종버전 (이 파일 기준) |


