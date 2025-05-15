import pandas as pd
from sqlalchemy import create_engine

# CSV 파일 리스트
files = [
    'tving_drama_all_merged.csv',
    'tving_entertainment_all_merged.csv',
    'tving_movie_all_merged.csv',
    'tving_애니_키즈.csv'
]

# CSV 병합
df_list = [pd.read_csv(f, encoding='utf-8') for f in files]
df = pd.concat(df_list, ignore_index=True)

# MySQL 연결 (encoding 인자 삭제)
engine = create_engine('mysql+pymysql://root:rubi@localhost:3306/ifitv_db')

# contents 테이블에 업로드
df.to_sql('contents', con=engine, if_exists='replace', index=False)

print("✅ MySQL 업로드 완료!")
