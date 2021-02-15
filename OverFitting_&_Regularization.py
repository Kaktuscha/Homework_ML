rmse_train, rmse_valid, avg_coef = [], [], []
for degree in DEGREES:
    results = cross_validate(make_model(degree), X, y, cv=5,  return_train_score=True, return_estimator=True, scoring='neg_root_mean_squared_error')
    rmse_train.append(-np.mean(results['train_score']))
    rmse_valid.append(-np.mean(results['test_score']))
    avg_coef.append( np.mean([np.mean(np.abs(model['reg'].coef_)) for model in results['estimator']   ]))

plot_fitting_graph(DEGREES, rmse_train, rmse_valid, xlabel='Complexity (degree)', ylabel='Error (RMSE)',  custom_metric=avg_coef, custom_label='avg(|$w_i$|)',  title='Least squares polynomial regression')