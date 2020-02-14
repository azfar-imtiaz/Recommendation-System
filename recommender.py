# from surprise import KNNWithMeans
from surprise import SVD


similarity_options = {
    'name': 'pearson',  # pearson correlation coefficient is centered cosine
    # setting this to False means we're computing similarities between items
    'user_based': False
}

# algo = KNNWithMeans(sim_options=similarity_options)
algo = SVD()
