from surprise import reader
from surprise import Dataset
import numpy as np
import pprint as pp
from surprise.model_selection import KFold, cross_validate, GridSearchCV, RandomizedSearchCV
from surprise.prediction_algorithms import NormalPredictor, BaselineOnly, \
    KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, \
    SVD, SVDpp, NMF, \
    SlopeOne, CoClustering

print('load file...')
# the reader parsers is created to parse the file containing the ratings
reader = reader.Reader(line_format="user item rating", sep=',', rating_scale=(1, 5), skip_lines=1)

# load_from_file is used to open the ratings.csv
data = Dataset.load_from_file('./dataset/ratings.csv', reader=reader)
print('ratings.csv file correctly loaded!', '\n')

########################################################
# 5FoldCV for all 11 Surpise Recommendation Algorithms #
########################################################


print('Start performing 5 Fold Cross Validation for each Algorithm...', '\n')
kf = KFold(n_splits=5, random_state=10, shuffle=True)  #cross validation object

predictors = [NormalPredictor(), BaselineOnly(),
              KNNBasic(), KNNWithMeans(), KNNWithZScore(), KNNBaseline(),
              SVD(), SVDpp(), NMF(),
              SlopeOne(), CoClustering()
             ]
mean_RMSE = list()
cv_results = dict()
for predictor in predictors:
    cv_result = cross_validate(predictor, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=4)  # execute 5-Fold CV
    mean_RMSE.append((str(predictor)[:-2], np.mean(cv_result['test_rmse']))) # store into mean_RMSE the Algortihm and its mean RMSE
    cv_results[str(predictor)[:-2]] = cv_result
mean_RMSE.sort(key=lambda x:x[1])  # sort tuples by using mean RMSE
print('\n', 'Sorted Recommendation Algorithms:', mean_RMSE, sep='\n')
for t in mean_RMSE:
    print(t[0], cv_results[t[0]], '\n', sep='\n')


###################################################
# SVD Hyperparamiter tuning by using GridSearchCV #
###################################################
print()
print('Starts performing Hyperparameter tuning for SVD...')
paramiters = {
    'n_factors':[60],  # number of factors to be considered
    'n_epochs': [70],  # number of iterations for the Stochastic Gradient Descent
    'init_mean': [0], #  mean of the normal distribution for factor vectors initialization
    'reg_all': [0.04, 0.09],  # regularization term for all parameters (reg_bu, reg_bi, reg_pu, reg_qi)
    # learning rate for the bu, bi, pu, qi paramiters
    'lr_bu': [0.0009,0.002],
    'lr_bi': [0.001,0.002,0.003],
    'lr_pu': [0.0009,0.002,0.005],
    'lr_qi': [0.0009,0.002,0.009],


}
grid = GridSearchCV(
    algo_class=SVD,  # algorithm class
    param_grid=paramiters,  # dictionary with all paramiters that have to be tuned
    measures=['RMSE'],  # evaluation metric
    cv=kf,  # crossvalidation object
    n_jobs=4,  # use 4 cores
    joblib_verbose=1000,

)
grid.fit(data)
print('\n', 'BEST SCORE: ' + str(grid.best_score['rmse']), 'BEST PARAMITERS: ' + str(grid.best_params), sep='\n')


### 5-Fold CV for SVD using best discovered parameteres
print()
print('Starts performing 5FoldCV by using the best found parameters for SVD...')
algo = SVD(n_factors=60,
           n_epochs=70,
           init_mean=0,
           reg_all=0.09,
           lr_bu=0.0009,
           lr_bi=0.002,
           lr_pu=0.005,
           lr_qi=0.009,
           random_state=91)
cross_validate(algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=4)


##################################################################
# KNNBaseline  Hyperparamiter tuning by using RandomizedSearchCV #
##################################################################
print()
print('Starts performing Hyperparameter tuning for KNNBaseline...')
paramiters2 = {
    'k' : [n for n in range(10,100,10)], # max number of neighbors
    'min_k' : [n for n in range(1,40,2)], # min number of neighbors
    'sim_options': {
        'name': ['cosine', 'msd', 'pearson', 'pearson_baseline'], # similarity measure
        'user_based': [True, False], # if the similarities will be computed between users or between items
        'min_support': [n for n in range(1,10,1)] #  min number of common items
    },    'bsl_options':{
        'method': ['sgd'], #estimation of baseline using Stochastic Gradient Descent
        'n_epochs':[n for n in range(20,80,10)], #  number of iteration for the chosen procedure (in this case SGD)
        'reg': [n for n in np.arange(0.02,0.1,0.01)], # regularization parameter of the cost function
        'learning_rate':[n for n in np.arange(0.002,0.01, 0.001)] # learning rate of SGD
    }
}
randgrid = RandomizedSearchCV(
    algo_class=KNNBaseline,
    param_distributions=paramiters2,
    n_iter=60,
    measures=['RMSE'],
    cv = kf,
    n_jobs=4, # use 4 cores
    joblib_verbose=1000
)

randgrid.fit(data)

print('BEST SCORE: '+str(randgrid.best_score['rmse']))
print('BEST PARAMITERS: ')
pp.pprint(randgrid.best_params)

### 5-Fold CV for KNNBaseline using best discovered parameteres
print()
print('Starts performing 5FoldCV by using the best found parameters for KNNBaseline...')

current_algo = KNNBaseline(k= 40,
                           min_k=11,
                           sim_options={
                               'name': 'pearson_baseline',
                               'user_based': False, # compute similarities between items
                               'min_support': 4
                           },
                           bsl_options={
                               'method': "sgd",
                               'learning_rate': 0.004,
                               'n_epochs': 50,
                               'reg': 0.08,
                           },
                           verbose=True)
cross_validate(current_algo, data, measures=['RMSE'], cv=kf, n_jobs=4, verbose=True)










