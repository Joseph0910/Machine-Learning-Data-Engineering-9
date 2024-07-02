
import dependencies_1
from dependencies_1 import * 

def model_classification_test(y_act,y_pred,thres):

	y_act = y_test2
	y_pred = logreg_predict_prob

	y_act = numpy.array(y_act)
	y_pred = numpy.array(y_pred)
	predicted = numpy.where(y_pred > 0.8, 1, 0 )
	tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_act,predicted).ravel()
	fnr = fn / (fn + tp)
	fpr = fp / (fp + tn)
	tpr = sklearn.metrics.recall_score(y_act,predicted)
	thresholds_2 = sklearn.metrics.roc_curve(y_act,predicted) # binary only 
	tnr = 1 - fpr 
	jaccard = sklearn.metrics.jaccard_score(y_act,predicted, average = 'macro')
	accuracy = sklearn.metrics.accuracy_score(y_act,predicted)
	ave_prec = sklearn.metrics.average_precision_score(y_act,predicted)
	prec = sklearn.metrics.precision_score(y_act,predicted)
	acc_bal = sklearn.metrics.balanced_accuracy_score(y_act,predicted)
	hamming = sklearn.metrics.hamming_loss(y_act,predicted)
	f1s_binary = sklearn.metrics.f1_score(y_act,predicted)
	rmse = sqrt(mean_squared_error(y_act,predicted))
	brier = sklearn.metrics.brier_score_loss(y_act,predicted)
	logloss = sklearn.metrics.log_loss(y_act,predicted)
	youden = (tpr + tnr) - 1
	roc_auc = sklearn.metrics.roc_auc_score(y_act,predicted)
	precision, recall,thresholds_1 = sklearn.metrics.precision_recall_curve(y_act,predicted) # binary only 
	kappa = sklearn.metrics.cohen_kappa_score(y_act, predicted) # 1 = complete agreement, otherwise chance agreement # 
	hinge_loss = sklearn.metrics.hinge_loss(y_act,predicted)
	matthew_corr = sklearn.metrics.matthews_corrcoef(y_act,predicted) # +1 perfect pred , 0 random, -1 inverse # 
	fbeta = sklearn.metrics.fbeta_score(y_act,predicted,0.5)

	columns = [fpr,tpr,fnr,tnr,jaccard,accuracy,prec,fbeta,ave_prec,acc_bal,hamming,f1s_binary,rmse,brier,logloss,youden,roc_auc,kappa,hinge_loss,matthew_corr]
	columns = pandas.DataFrame(columns)

	rows = ['False Positive Rate','True Positive Rate','False Negative Rate','True Negative Rate','Jaccard Score','Accuracy Rate',
			'Precision','F Beta Score','Ave Precision','Ave Precision Rate','Hamming Loss','F1 Score (Binary)','RMSE','Brier Score', 
			'Log Loss (Logistic)','Youden','ROC AUC Score','Cohen Kappa Score','Hinge Loss','Matthew Correlation Coeff']

	rows = pandas.DataFrame(rows)
	performance_analysis = pandas.concat([rows,columns],axis = 1,sort=False)
	    
	return performance_analysis



def model_rank_test(y_act,y_pred):

	coverage = coverage_error(y_act,y_pred)
	rank_prec_score = label_ranking_average_prediction_score(y_act,y_pred)
	rank_loss = label_ranking_loss(y_act,y_pred)
	dcg_score_v = dcg_score(y_act,y_pred)

	columns = [coverage,rank_prec_score,rank_loss,dcg_score_v]
	columns = pandas.DataFrame(columns)

	rows = ['Coverage','Rank Precision Score', 'Rank Loss', 'Discounted Cum. Gain']

	rows = pandas.DataFrame(rows)
	performance_analysis = pandas.concat([rows,columns],axis = 1,sort=False)
	    
	return performance_analysis


def model_regression_test(y_true,y_pred):

	y_true = numpy.array(y_true)
	y_pred = numpy.array(y_pred)

	exp_var = sklearn.metrics.explained_variance_score(y_true,y_pred)
	max_error_ = sklearn.metrics.max_error(y_true,y_pred)
	mean_abs_error = sklearn.metrics.mean_absolute_error(y_true,y_pred)
	mean_sq_error = sklearn.metrics.mean_squared_error(y_true,y_pred)
#	mean_sq_log_error = sklearn.metrics.mean_squared_log_error(y_true,y_pred)  can only be used for positive integer targets 
	med_abs_error = sklearn.metrics.median_absolute_error(y_true,y_pred)
	r2_value = sklearn.metrics.r2_score(y_true,y_pred)

	columns = [exp_var,max_error_,mean_abs_error,mean_sq_error,med_abs_error,r2_value]
	columns = pandas.DataFrame(columns)
	
	rows = ['Exp Variance','Max Error','Mean Abs. Error','Mean Squared Error' ,'Median Abs Error','R2 Score']

	rows = pandas.DataFrame(rows)
	
	performance_analysis = pandas.concat([rows,columns],axis = 1,sort=False)
	    
	return performance_analysis


def cluster_valid_test(y_true,y_pred):

	adj_mut_info = adjusted_mutual_info_score(y_true,y_pred)
	adj_rand_score = adjusted_rand_score(y_true,y_pred)
	complete_score = completeness_score(y_true,y_pred)
	fowlkes_mall_score = fowlkes_mallows_score(y_true,y_pred)
	homo_score = homogeneity_score(y_true,y_pred)
	mut_info_score = mutual_info_score(y_true,y_pred)
	norm_mut_info_score = normalized_mutual_info_score(y_true,y_pred)
	v_measure = v_measure_score(y_true,y_pred)
	
	columns = [adj_mut_info,adj_rand_score,complete_score,fowlkes_mall_score,
				homo_score,mut_info_score,norm_mut_info_score,v_measure]

	columns = pandas.DataFrame(columns)
	
	rows = ['Adj. Mutual Info Score','Adj. Rand Score','Completeness Score','Fowlkes Mallows Score',
			'Homogeneity Score','Mutual Info Score','Normalized Mutual Info Score', 'V Measure Score']

	rows = pandas.DataFrame(rows)
	
	performance_analysis = pandas.concat([rows,columns],axis = 1,sort=False)
	    
	return performance_analysis



def cross_valid_data(model,X,y,cv_ind):

	cv_results = cross_validate(model, X, y, cv= cv_ind) # 3 fold cv #
	results = sorted(cv_results.keys())

	cross_valid_score = cross_val_score(model,X,y, cv = cv_ind) # 4 fold cross validation 
		
	return results, cross_valid_score


def dummy_classifier(X_train,y_train,X_test,y_test,const_value):

	clf_most_freq = DummyClassifier(strategy='most_frequent', random_state=910)
	clf_most_freq.fit(X_train, y_train)
	score_freq = clf_most_freq.score(X_test, y_test)

	clf_strat = DummyClassifier(strategy='stratified', random_state=910)
	clf_strat.fit(X_train, y_train)
	score_strat = clf_strat.score(X_test, y_test)

	clf_prior = DummyClassifier(strategy='prior', random_state=910)
	clf_prior.fit(X_train, y_train)
	score_prior = clf_prior.score(X_test, y_test)

	clf_uni = DummyClassifier(strategy='uniform', random_state=910)
	clf_uni.fit(X_train, y_train)
	score_uni = clf_uni.score(X_test, y_test)

	clf_constant = DummyClassifier(strategy='constant', random_state=910, constant = const_value)
	clf_constant.fit(X_train, y_train)
	score_con = clf_constant.score(X_test, y_test)

	columns = [score_freq, score_strat, score_prior, score_uni, score_con]

	columns = pandas.DataFrame(columns)
	
	rows = ['Dummy Classifier: Most Freq Label', 'Dummy Classifier: Stratified to Class Dist.',
			'Dummy Classifier: Based on Max. Class Prior', 'Dummy Classifier: Uniform',
			'Dummy Classifier: Constant Label']

	rows = pandas.DataFrame(rows)
	
	performance_analysis = pandas.concat([rows,columns],axis = 1,sort=False)
	    
	return performance_analysis


def dummy_regressor(X,Y,quantile_value,const_value):


	model1 = DummyRegressor(strategy= "mean")
	model1.fit(X,Y)
	model1.predict(X)
	score1 = model1.score(X,Y)

	model2 = DummyRegressor(strategy = "median")
	model2.fit(X,Y)
	model2.predict(X)
	score2 = model2.score(X,Y)

	model3 = DummyRegressor(strategy = "quantile", quantile = quantile_value)
	model3.fit(X,Y)
	model3.predict(X)
	score3 = model3.score(X,Y)

	model4 = DummyRegressor(strategy = "constant", constant = const_value)
	model4.fit(X,Y)
	model4.predict(X)
	score4 = model4.score(X,Y)

	columns = [score1, score2, score3, score4 ]

	columns = pandas.DataFrame(columns)
	
	rows = ['Dummy Classifier: Average', 'Dummy Classifier: Median',
			'Dummy Classifier: Specified Quantile', 'Dummy Classifier: Constant Value']

	rows = pandas.DataFrame(rows)
	
	performance_analysis = pandas.concat([rows,columns],axis = 1,sort=False)
	    
	return performance_analysis


def hyperparamater_test(model,X,Y,hyper_param_name,cross_val_fold):

	X = numpy.array(X)
	Y = numpy.array(Y)
	param_range = numpy.logspace(-6, -1, 5)

	train_scores, test_scores = validation_curve(model, X, Y, param_name = hyper_param_name, cv= cross_val_fold)

	train_scores_mean = numpy.mean(train_scores, axis=1)
	train_scores_std = numpy.std(train_scores, axis=1)
	test_scores_mean = numpy.mean(test_scores, axis=1)
	test_scores_std = numpy.std(test_scores, axis=1)


	plt.title("Validation Curve")
	plt.xlabel(hyper_param_name)
	plt.ylabel("Score")
	plt.ylim(0.0, 1.1)
	lw = 2
	plt.semilogx(param_range, train_scores_mean, label="Training score",
	             color="darkorange", lw=lw)
	plt.fill_between(param_range, train_scores_mean - train_scores_std,
	                 train_scores_mean + train_scores_std, alpha=0.2,
	                 color="darkorange", lw=lw)
	plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
	             color="navy", lw=lw)
	plt.fill_between(param_range, test_scores_mean - test_scores_std,
	                 test_scores_mean + test_scores_std, alpha=0.2,
	                 color="navy", lw=lw)
	plt.legend(loc="best")
	plt.show()


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=numpy.linspace(.1, 1.0, 5)):

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)
    fit_times_mean = numpy.mean(fit_times, axis=1)
    fit_times_std = numpy.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


# Future Work #
# scoring strategy # https://scikit-learn.org/stable/modules/model_evaluation.html#  

