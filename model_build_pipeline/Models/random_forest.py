from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def model(train_x, train_y, cv_folds, scoring):
    seed = 1
    lr = RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight={0:0.8, 1:1.4})
    param_grid = {
        'n_estimators':[50,100,150,200,250],
        'max_depth':[5,7,10,15,None],
        'max_features':['sqrt',10,None],
        'class_weight':[{0:0.8, 1:1.2}, {0:0.8, 1:1.4}]
    }

    lr_grid = GridSearchCV(estimator=lr,
                        n_jobs=-1,
                        refit=True,
                        param_grid=param_grid,
                        scoring=scoring,
                        cv=cv_folds,
                        return_train_score=False)

    print('Training RF model...')
    lr_grid.fit(train_x, train_y)

    return lr_grid