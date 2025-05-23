
# Lightgbm model - sklearn API


# regressor

classlightgbm.LGBMRegressor(*, 
                            boosting_type='gbdt', 
                            num_leaves=31, 
                            max_depth=-1, 
                            learning_rate=0.1, 
                            n_estimators=100, 
                            subsample_for_bin=200000, 
                            objective=None, 
                            class_weight=None, 
                            min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, 
                            subsample=1.0, subsample_freq=0, 
                            colsample_bytree=1.0, 
                            reg_alpha=0.0, reg_lambda=0.0, 
                            random_state=None, 
                            n_jobs=None, 
                            importance_type='split', 
                            **kwargs)
# training

fit(X, y, sample_weight=None, 
    init_score=None, 
    eval_set=None, eval_names=None, eval_sample_weight=None, eval_init_score=None, eval_metric=None, 
    feature_name='auto', categorical_feature='auto', 
    callbacks=None, init_model=None
   )
# inference
predict(X, 
        raw_score=False, 
        start_iteration=0, 
        num_iteration=None, 
        pred_leaf=False, 
        pred_contrib=False, 
        validate_features=False, 
        **kwargs)
