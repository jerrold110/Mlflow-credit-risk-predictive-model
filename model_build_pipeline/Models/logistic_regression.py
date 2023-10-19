from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def model(train_x, train_y, cv_folds, scoring):
    seed = 1
    lr = LogisticRegression(random_state=seed, max_iter=200)

    reg_str = [10,4,2,1,.66,.5] # inverse (1/10...)
    param_grid = [
        {
            'penalty':[None],
            'solver':['lbfgs'],
            'C':reg_str
        },
        {
            'penalty':['l2'],
            'solver':['lbfgs'],
            'C':reg_str
        },
        {
            'penalty':['l1'],
            'solver':['saga'],
            'C':reg_str
        },
        # for elasticnet, set l1 ratio, and C? value of C should apply to a given ratio: 0.1-0.9
        {
            'penalty':['elasticnet'],
            'solver':['saga'],
            'C':reg_str,
            'l1_ratio':[.2, .5, .7, .9, .95]
        }
    ]

    lr_grid = GridSearchCV(estimator=lr,
                        n_jobs=-1,
                        refit=True,
                        param_grid=param_grid,
                        scoring=scoring,
                        cv=cv_folds,
                        return_train_score=False)

    print('Training LR model...')
    lr_grid.fit(train_x, train_y)

    return lr_grid