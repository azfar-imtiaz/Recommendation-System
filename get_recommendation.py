from surprise.accuracy import rmse
from surprise.model_selection import train_test_split

# from load_data import data
from load_data import movie_lens_data
from recommender import algo


# print("Building full train set...")
# training_set = movie_lens_data.build_full_trainset()
print("Building train test split...")
training_set, testing_set = train_test_split(movie_lens_data, test_size=0.2)

print("Fitting model to data...")
algo.fit(training_set)

print("Getting predictions on testing data...")
predictions = algo.test(testing_set)

rmse_score = rmse(predictions)
print("Root mean squared error on predictions is: {}".format(rmse_score))

print("Getting prediction for user {} on movie {}".format('E', 2))
prediction = algo.predict('E', 2)
print(prediction.est)
