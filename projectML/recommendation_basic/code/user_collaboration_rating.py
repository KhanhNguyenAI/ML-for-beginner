import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import re
# Đọc dữ liệu
df = pd.read_csv(r'C:\Users\97ngu\OneDrive\Desktop\course\ML-EX\chua lam\Recommendation_System\movie_data\ratings.csv', encoding='latin-1', sep='\t')
movies = pd.read_csv(r'C:\Users\97ngu\OneDrive\Desktop\course\ML-EX\chua lam\Recommendation_System\movie_data\movies.csv', encoding='latin-1', sep='\t')
# Tạo pivot table: hàng là movieId, cột là userId, giá trị là rating
pivot = df.pivot_table(index='movie_id', columns='user_id', values='rating')

# Tính cosine similarity giữa các user
user_sim = cosine_similarity(pivot.T.fillna(0))
user_sim_df = pd.DataFrame(user_sim, index=pivot.columns, columns=pivot.columns)
# print(user_sim_df)
# Hàm dự đoán rating cho user 1 với các phim chưa rating
def predict_rating(user_id, movie_id):
    # Lấy các user đã rating phim này
    users_rated = pivot.loc[movie_id][pivot.loc[movie_id].notnull()].index
    # Lấy similarity giữa user_id và các user đã rating
    sims = user_sim_df.loc[user_id, users_rated]
    ratings = pivot.loc[movie_id, users_rated]
    if sims.sum() == 0:
        return np.nan
    # Dự đoán rating bằng weighted average
    return np.dot(sims, ratings) / sims.sum()

# Điền rating dự đoán cho user 1 vào các phim chưa rating
user_id = 1


for movie_id in pivot.index[pivot[user_id].isnull()]:
    pivot.at[movie_id, user_id] = predict_rating(user_id, movie_id)

top_n = 10 
# print("Các rating đã được dự đoán cho user 1:")
# print(pivot[user_id][pivot[user_id].notnull()])
pivot[user_id] = pivot[user_id][pivot[user_id].notnull()]

# In ra 10 rating cao nhất mà user 1 đã được dự đoán
predicted_ratings = pivot[user_id][df['movie_id'].unique()]
predicted_ratings = predicted_ratings[predicted_ratings.notnull()]  # loại bỏ NaN
top_10 = predicted_ratings.sort_values(ascending=False).head(10)
# Lấy top 10 movie_id và rating
print(top_10)
top_10_df = top_10.reset_index().rename(columns={'index': 'movie_id', user_id: 'predicted_rating'})
print(top_10_df)
# Gộp với bảng movies để lấy tên phim
top_10_with_titles = pd.merge(top_10_df, movies[['movie_id', 'title']], on='movie_id', how='left')

# Thêm cột year bằng regex
top_10_with_titles['year'] = top_10_with_titles['title'].str.extract(r'\((\d{4})\)').astype(float)

# Sắp xếp theo year tăng dần
top_10_sorted = top_10_with_titles.sort_values('year',ascending=False)

print("Top 10 phim được dự đoán rating cao nhất cho user 1 (sắp xếp theo năm):")
print(top_10_sorted[['title']].to_string(index=False))


'''

● Weighted Hybrid: Kết hợp điểm số từ các hệ thống khác nhau 
bằng cách gán trọng số. 
● Switching Hybrid: Chọn thuật toán phù hợp nhất dựa trên ngữ 
cảnh. 
● Mixed Hybrid: Trình bày gợi ý từ các hệ thống khác nhau cạnh 
nhau. 
● Feature Combination: Kết hợp các đặc trưng (features) từ cả hai 
nguồn trước khi áp dụng một mô hình. 
● Meta-level: Một thuật toán học cách gợi ý của một thuật toán 
khác. '''