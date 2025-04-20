# Simulating execution in Google Colab

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Sample Data ---
# Let's imagine we have data like this: (user_id, item_id, rating)
# A higher rating means the user liked the item more.
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    'item_id': ['item_A', 'item_B', 'item_D', 'item_A', 'item_C', 'item_B', 'item_C', 'item_D', 'item_A', 'item_C', 'item_D', 'item_B', 'item_E'],
    'rating':  [5, 4, 3, 4, 5, 5, 4, 2, 3, 5, 4, 4, 5]
}
df = pd.DataFrame(data)

print("--- Original Interaction Data ---")
print(df)
print("\n")

# --- 2. Create User-Item Matrix ---
# We'll pivot the table so:
# - Rows are users
# - Columns are items
# - Values are ratings
# 'NaN' (Not a Number) means the user hasn't rated that item. We fill these with 0.
user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating')
user_item_matrix = user_item_matrix.fillna(0) # Fill missing ratings with 0

print("--- User-Item Matrix ---")
print(user_item_matrix)
print("\n")

# --- 3. Calculate Item-Item Similarity ---
# We use cosine similarity. It measures the cosine of the angle between two vectors.
# In our case, the vectors are the columns (items) in the user-item matrix.
# It tells us how similar items are based on user ratings.
# Important: cosine_similarity expects samples (items in our case) as ROWS,
# so we need to transpose (.T) the matrix before calculating similarity.
item_similarity = cosine_similarity(user_item_matrix.T)

# Convert the similarity results into a DataFrame for easier understanding
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

print("--- Item-Item Similarity Matrix ---")
print(item_similarity_df)
print("\n")

# --- 4. Make Recommendations for a User ---
def get_recommendations(user_id, user_item_matrix, item_similarity_df, num_recommendations=2):
    """
    Generates recommendations for a specific user.
    """
    print(f"--- Generating Recommendations for User {user_id} ---")

    # Get ratings for the target user
    user_ratings = user_item_matrix.loc[user_id]

    # Find items the user has NOT interacted with (rating == 0)
    items_to_predict = user_ratings[user_ratings == 0].index

    # Dictionary to store predicted scores for potential recommendations
    recommendation_scores = {}

    # Iterate through items the user hasn't rated
    for item_to_predict in items_to_predict:
        total_similarity = 0
        weighted_sum = 0

        # Iterate through items the user HAS rated
        for rated_item, rating in user_ratings[user_ratings > 0].items():
            # Get the similarity between the item we want to predict and the item the user already rated
            similarity = item_similarity_df.loc[item_to_predict, rated_item]

            # Only consider positive similarity
            if similarity > 0:
                weighted_sum += similarity * rating # Weight the user's rating by the similarity
                total_similarity += similarity     # Keep track of the total similarity for normalization

        # Calculate the predicted score (avoid division by zero)
        if total_similarity > 0:
            predicted_score = weighted_sum / total_similarity
            recommendation_scores[item_to_predict] = predicted_score

    # Sort recommendations by score in descending order
    sorted_recommendations = sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)

    # Return the top N recommendations
    return sorted_recommendations[:num_recommendations]

# --- Example Usage ---
# Let's get recommendations for User 1
user_to_recommend_for = 1
recommendations = get_recommendations(user_to_recommend_for, user_item_matrix, item_similarity_df, num_recommendations=3)

print(f"\nTop recommendations for User {user_to_recommend_for}:")
if recommendations:
    for item, score in recommendations:
        print(f"- Item: {item}, Predicted Score: {score:.4f}")
else:
    print("No recommendations found.")

# Let's get recommendations for User 5
user_to_recommend_for = 5
recommendations = get_recommendations(user_to_recommend_for, user_item_matrix, item_similarity_df, num_recommendations=3)

print(f"\nTop recommendations for User {user_to_recommend_for}:")
if recommendations:
    for item, score in recommendations:
        print(f"- Item: {item}, Predicted Score: {score:.4f}")
else:
    print("No recommendations found.")

'''
--- Original Interaction Data ---
    user_id item_id  rating
0         1  item_A       5
1         1  item_B       4
2         1  item_D       3
3         2  item_A       4
4         2  item_C       5
5         3  item_B       5
6         3  item_C       4
7         3  item_D       2
8         4  item_A       3
9         4  item_C       5
10        4  item_D       4
11        5  item_B       4
12        5  item_E       5


--- User-Item Matrix ---
item_id  item_A  item_B  item_C  item_D  item_E
user_id
1           5.0     4.0     0.0     3.0     0.0
2           4.0     0.0     5.0     0.0     0.0
3           0.0     5.0     4.0     2.0     0.0
4           3.0     0.0     5.0     4.0     0.0
5           0.0     4.0     0.0     0.0     5.0


--- Item-Item Similarity Matrix ---
item_id    item_A    item_B    item_C    item_D    item_E
item_id
item_A   1.000000  0.365148  0.670820  0.707107  0.000000
item_B   0.365148  1.000000  0.458831  0.316228  0.768221
item_C   0.670820  0.458831  1.000000  0.715542  0.000000
item_D   0.707107  0.316228  0.715542  1.000000  0.000000
item_E   0.000000  0.768221  0.000000  0.000000  1.000000


--- Generating Recommendations for User 1 ---

Top recommendations for User 1:
- Item: item_C, Predicted Score: 4.1168

--- Generating Recommendations for User 5 ---

Top recommendations for User 5:
- Item: item_A, Predicted Score: 4.0000
- Item: item_C, Predicted Score: 4.0000
- Item: item_D, Predicted Score: 4.0000
'''