
## Model Validation ## 
seed(10)
    batch2 = int(len(test_x)*batch_num)
    pred_test = model.predict(test_x,batch_size = batch2, verbose = 2)
    pred_test = pandas.DataFrame(pred_test,columns = columns_y2)
    test_x = pandas.DataFrame.reset_index(test_x)
    test_x = test_x.drop(columns = 'index')

    ## Merge Prediction to dataset ## 
    test_x['Probability'] = pred_test  
    max_prob = round(max(test_x.Probability),2)

    ## Assign a new field for the actual and predicted output ## 
    actual = test_y
    pred_p = test_x.Probability

    def binary_conversion(output):
    	for i in test_x.Probability:
    		if i > 0.90:
    			output.append(1)
    		else: 
    			output.append(0)

    predicted = []
    binary_conversion(predicted)
    predicted = pandas.DataFrame(predicted)

    ## Confusion Matrix ## 
    tn, fp, fn, tp = confusion_matrix(actual,predicted).ravel()

    ## False Positive Rate ## 
    fpr = fp / (tn + fp)

    ## True Positive ## 
    rec = recall_score(actual,predicted)

    ## False Negative Rate ## 
    fnr = fn / (fn + tp)

    ## True Negative Rate ## 
    tnr = 1 - fpr 

    ## Misclassification Rate ## 
    misclass = 1 - accuracy_score(actual,predicted)

    ## F1 Score ##
    f1s = f1_score(actual,predicted)

    ## RMSE ##
    rmse = sqrt(mean_squared_error(actual,predicted))

    ## Youden ## 
    youden = (rec + tnr) - 1

    ## Gini Score ##

    def Gini(y_true, y_pred):
        # check and get number of samples
        assert y_true.shape == y_pred.shape
        n_samples = y_true.shape[0]
        
        # sort rows on prediction column 
        # (from largest to smallest)
        arr = numpy.array([y_true, y_pred]).transpose()
        true_order = arr[arr[:,0].argsort()][::-1,0]
        pred_order = arr[arr[:,1].argsort()][::-1,0]
        
        # get Lorenz curves
        L_true = numpy.cumsum(true_order) / numpy.sum(true_order)
        L_pred = numpy.cumsum(pred_order) / numpy.sum(pred_order)
        L_ones = numpy.linspace(1/n_samples, 1, n_samples)
        
        # get Gini coefficients (area between curves)
        G_true = numpy.sum(L_ones - L_true)
        G_pred = numpy.sum(L_ones - L_pred)
        
        # normalize to true Gini coefficient
        return G_pred/G_true

    gini = Gini(actual,pred_p)

    ## Columns ## 
    performance = [gini,fpr,rec,fnr,tnr,misclass,f1s,rmse,youden,train_pct,batch_num,dropout,num_hidden,num_output,num_epoch]

    ## Convert to Dataframe ##
    performance = pandas.DataFrame(performance)

    ## Row Naming ## 
    rows = ['Gini','False Positive Rate','True Positive Rate', 'False Negative Rate', 
            'True Negative Rate','Misclassification Rate','F1 Score',
            'RMSE','Youden','Train Percent','Batch Percent','Dropout Percent','Hidden Layers','Output Layers','Epochs']

    rows = pandas.DataFrame(rows)

    performance_analysis = pandas.concat([rows,performance],axis = 1,sort=False)
    print(performance_analysis)

    # Optimize Model Strength, Error, and Classification # 
    if gini > 0.8 and misclass < 0.20 and rec > 0.90:
        performance_analysis.to_csv(r'/Users/josephgalea/Documents/Environment1/Model Evaluation/Model_Validation_Metrics_TPR_%s.csv' %(round(rec,4)))
        ## File Naming ##
        filename = 'model_%s.sav' %(rec)
        pickle.dump(model,open(filename,'wb'))