
# Collaborative Filtering using Surprise SVD
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

# Sample data: userId, itemId, rating
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.25)

# Train SVD model
model = SVD()
model.fit(trainset)
predictions = model.test(testset)

# Evaluate RMSE
print("RMSE:", accuracy.rmse(predictions))
