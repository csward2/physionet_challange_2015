import numpy as np
import multiprocessing as mp
# import xml.etree.ElementTree as ET
from sys import getsizeof
import os
from sklearn import cross_validation, linear_model, metrics, ensemble, svm, grid_search
from datetime import datetime
import pickle
import cPickle
import json,random
import scipy.io

from feature_extract import *
# from edf_parse import UWSleep_parse_data
# import matplotlib.pyplot as plt




class Event:
    def __init__(self, start, duration):
        self.start = start 
        self.duration = duration


def train_model(train_data, train_labels):
    '''
    Averaged ElasticNet SGD SVM 
        non-default parameters:
        set n_jobs = -1 to use all available CPU
        use hinge loss for linear SVM
        verbose to 1 for chatty interface
        average to True for ASGD
        elasticnet regularization with 25 percent L1 and 75 percent L2 penalty
        set class_weights to auto for running all training data without downsampling negatives

    Logistic Ridge Regression 
        non-default parameters:
        params same as Averaged ElasticNet SGD SVM execpt
        use log loss for logistic regression
        ridge (L2) regularization

    Random Forest 
        non-default parameters:
        set n_jobs = -1 to use all available CPU
        entropy for split evaluation on information gain
        20 decision trees
        set max_features to none
    '''
    n_workers = -1
    if classifier_type == classifier_options[0]:
        clf = linear_model.SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1,
            eta0=0.0, fit_intercept=True, l1_ratio=0.25,
            learning_rate='optimal', loss='hinge', n_iter=30, n_jobs=n_workers,
            penalty='elasticnet', power_t=0.5, random_state=None, shuffle=True,
            verbose=1, warm_start=False, average = True)

    if classifier_type == classifier_options[1]:
        clf = linear_model.SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1,
            eta0=0.0, fit_intercept=True, l1_ratio=0.25,
            learning_rate='optimal', loss='log', n_iter=20, n_jobs=n_workers,
            penalty='l2', power_t=0.5, random_state=None, shuffle=True,
            verbose=1, warm_start=False, average = True)

    if classifier_type == classifier_options[2]:
        if grid_search_toggle == True:
            max_depth = [20,50,100]
            tuned_parameters = {'max_depth':max_depth}

            rf = ensemble.RandomForestClassifier(n_estimators=20, criterion='entropy', #max_depth=None, 
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                max_features=None, max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=n_workers, 
                random_state=None, verbose=1, warm_start=False, class_weight=None)
            clf = grid_search.GridSearchCV(rf, tuned_parameters, cv=5, n_jobs=n_workers, verbose=0)
        else:
            max_d = 20
            clf = ensemble.RandomForestClassifier(n_estimators=25, criterion='entropy', max_depth=max_d, 
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                max_features=None, max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=n_workers, 
                random_state=None, verbose=1, warm_start=False, class_weight=None)

    if classifier_type == classifier_options[3]:
        if grid_search_toggle == True:
            C_range = np.logspace(-2, 10, 13)
            gamma_range = np.logspace(-9, 3, 13)
            tuned_parameters = { 'gamma': gamma_range, 'C': C_range }
            svc = svm.SVC(kernel='rbf', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=50000, 
                class_weight=None, verbose=True, max_iter=-1, random_state=None)
            clf = grid_search.GridSearchCV(svc, tuned_parameters, cv=5, n_jobs=n_workers, verbose=0)
        else: 
            clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, 
                probability=True, tol=0.001, cache_size=50000, class_weight=None, verbose=False, 
                max_iter=-1, random_state=None)

    # Train model using all train_data
    clf.fit(train_data, train_labels)

    if grid_search_toggle == True:
        if classifier_type == classifier_options[3] or classifier_type == classifier_options[2]:
            # for params, mean_score, scores in clf.grid_scores_:
            #     print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
            #     print
            #     print 'Best params: ' + clf.best_params_
            #     print
            all_results.append(clf.grid_scores_)
            all_results.append(clf.best_params_)

    # from sklearn.externals import joblib
    # modelname = 'rf_120.p'
    # joblib.dump(clf, modelname) 

    return clf

def test_model(clf, test_data, test_labels):
    # Predict using model all test_data
    pred_labels = clf.predict(test_data)

    # generate classification report and total accuracy percentage
    class_report = metrics.classification_report(test_labels, pred_labels)
    tot_accuracy = metrics.accuracy_score(test_labels, pred_labels)
    pred_probs = clf.predict_proba(test_data)

    # return classification report and accuracy as results
    results = []
    results.append(tot_accuracy)    # results[0]
    results.append(pred_probs)      # results[1]
    results.append(test_labels)     # results[2]
    results.append(class_report)    # results[3]

    return results

def train_and_test(learning_lst):
    X_train = learning_lst[0]
    X_test = learning_lst[1]
    y_train = learning_lst[2]
    y_test = learning_lst[3]

    ml_model = train_model(X_train, y_train)
    results = test_model(ml_model, X_test, y_test)

    print "Accuracy Score: ", results[0]
    print
    print "Classification Report: "
    print results[1]
    print

    print "Model"
    print ml_model
    with open('/mnt/my-data/final/' + 'model_500.pickle', 'wb') as handle:
        pickle.dump(ml_model, handle)

    return results

def parse_and_extract(patientID):
    global count
    global total_windows_subprocess

    patient_data = parse_data(patientID)
    start = datetime.now()
    features = extract_all_features(patient_data)
    end = datetime.now()

    num_events = 1
    count+=1
    total_windows_subprocess+=1

    print "ID: " + patientID[0] + " ---  P&E Time: " + str(end-start) + " --- # Events: " + str(num_events)
    print "Subprocess Patient Count: " + str(count) + " --- # Cumulative Windows: " + str(total_windows_subprocess)
    print
    return features






# Regular batch execution of Classifier
if __name__ == '__main__' and True:
    
    count = 0
    total_windows_subprocess = 0

    # read local data files if debug is true
    debug = True
    # paralellize all computations if true
    multithread = False
    # do parse_and_extract if true
    p_and_e = True
    # do classification if true
    ml_classify = True
    # do gridsearch if true
    grid_search_toggle = False
    # number of patients
    num_patients = 10

    # current_b = 'b'+str(datetime.now())+'.p'
    # current_A = 'A'+str(datetime.now())+'.p'
    current_b = 'b_test_final.p'
    current_A = 'A_test_final.p'
    current_window = 'Window_test_final.p'
    current_header = 'Headers_test_final.p'

    ml_model = None


    #### the type of event we are classifying ####
    event_type = 'SDO:ObstructiveApneaFinding'
    # event_type = 'SDO:HypopneaFinding'

    #### a list of the classifiers we wish to evaluate ####
    classifier_options = ['Averaged ElasticNet SGD SVM', 'Logistic Ridge Regression', 'Random Forest', 'GridSearch Kernel SVM']
    #classifier_options = ['Random Forest']

    if debug == True: 
        # PATHVAR = '/home/ubuntu/ensodata/sample_data/'
        PATHVAR = '/Users/samrusk/Coding/physionet_challange_2015/entry/training/'
        LISTVAR = PATHVAR + 'ALARMS'
    else:
        # PATHVAR = '/mnt/mydata/ensodata/sample_data/'
        PATHVAR = '/home/ubuntu/ensodata/sample_data/'
        LISTVAR = PATHVAR + 'UWSleep_filenames.txt'


    # open and import list of patients
    # f = open(LISTVAR)
    full_patient_list = []
    with open(LISTVAR, "rt") as f:
        for line in f:
            values = line.split(',')
            values[0] = PATHVAR + values[0] + '.mat'
            values[2] = int(values[2][:-1])
            full_patient_list.append(values)




    num_patients = 3

    if p_and_e == True:
        patient_list = random.sample(full_patient_list,num_patients)

        if multithread == True:
            worker_count = mp.cpu_count()
            pool = mp.Pool(processes=worker_count)    
            print "Starting Pool" 
            print patient_list
            print "Number of Patients: " + str(len(patient_list))
            event_training_matrix = pool.map(parse_and_extract,patient_list,chunksize=1)

            pool.close()
            pool.join()
            print "Ending Join"
        else:
            event_training_matrix = []
            # for patient_id in patient_list[0:50]:
            for patient_id in patient_list:
                patient_feature = parse_and_extract(patient_id)
                event_training_matrix.append(patient_feature)
                print np.shape(event_training_matrix),np.shape(patient_feature)

        print event_training_matrix



    # for num_patients in num_list:
    #     if p_and_e == True:
    #         # open and import list of patients
    #         print "patient list" + str(len(full_patient_list))
    #         patient_list = random.sample(full_patient_list,num_patients)

    #         # determine number of workers          
    #         # run parse_data on patients in patient_list asynchronously             
    #         # run extract features on patient_matrix of events and signals
    #         if multithread == True:

    #             worker_count = mp.cpu_count() - 4
    #             pool = mp.Pool(processes=worker_count)    
    #             print "Starting Pool" 
    #             print patient_list
    #             print "Number of Patients: " + str(len(patient_list))
    #             event_training_matrix = pool.map(parse_and_extract,patient_list,chunksize=1)

    #             pool.close()
    #             pool.join()
    #             print "Ending Join"
    #         else:
    #             event_training_matrix = []

    #             # for patient_id in patient_list[0:50]:
    #             for patient_id in patient_list:
    #                 patient_feature = parse_and_extract(patient_id)
    #                 event_training_matrix.append(patient_feature)
    #                 print np.shape(event_training_matrix),np.shape(patient_feature)

    #         A = []
    #         b = []
    #         windows = []
    #         headers = []

    #         for ii, patient in enumerate(event_training_matrix):
    #             # Check for bad Files and None's thrown earlier
    #             if patient == None:
    #                 print "\nSkipping Patient #" + str(ii)
    #                 continue


    #             A_temp = patient['A']
    #             b_temp = patient['b']
    #             windows_temp = patient['windows']
    #             headers = patient['headers']


    #             tmp = []
    #             for jj in xrange(0,len(A_temp)):
    #                 tmp.append([x if not np.isnan(x) else 0 for x in A_temp[jj]])
                
    #             A.append([tmp,windows_temp])
    #             # A.append([tmp,0])
    #             b.append(b_temp)
    #             windows.append(windows_temp)


    #         print "Saving Pickles!"

    #         # cPickle.dump(b, open( '/mnt/mydata/final/'+current_b,"wb"))
    #         # cPickle.dump(A, open( '/mnt/mydata/final/'+current_A,"wb"))
    #         # cPickle.dump(windows, open( '/mnt/mydata/final/'+current_window,"wb"))
    #         # cPickle.dump(headers, open( '/mnt/mydata/final/'+current_header,"wb"))

    #         print (np.shape(A))
    #         print (np.shape(b))

    #     else:
    #         print "Opening Pickles!"
    #         # b = cPickle.load( open( '/mnt/mydata/final/'+current_b, "rb" ))
    #         # A = cPickle.load( open( '/mnt/mydata/final/'+current_A, "rb" ))
    #         # windows = cPickle.load( open( '/mnt/mydata/final/'+current_window, "rb" ))
    #         # headers = cPickle.load( open( '/mnt/mydata/final/'+current_header, "rb" ))


    #         print np.shape(A), np.shape(b)


    #     multithread = True
    #     if ml_classify == True:
    #         num_cv_folds = 10
    #         classifier_type = classifier_options[2]

    #         # for kk in range(2,4):
    #         #     classifier_type = classifier_options[kk]
    #         average_accuracy = 0

    #         all_results = []
    #         best_model = 0
    #         best_accuracy = 0
    #         for i in range(0,num_cv_folds):
    #             # Set CV holdout proportion
    #             print "shape",np.shape(A),np.shape(b),np.shape(A[0]),np.shape(A[0][0])
    #             X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = cross_validation.train_test_split(A, b, test_size=0.1, random_state=i)

    #             X_train = []
    #             X_test = []
    #             y_train = []
    #             y_test = []
    #             window_test = []

    #             for j in range(0,len(X_train_tmp)):
    #                 X_train += X_train_tmp[j][0]
    #                 y_train += y_train_tmp[j]

    #             for j in range(0,len(X_test_tmp)):
    #                 X_test += X_test_tmp[j][0]
    #                 y_test += y_test_tmp[j]
    #                 window_test += X_test_tmp[j][1]

    #             ml_model = train_model(X_train, y_train)
    #             results = test_model(ml_model, X_test, y_test)


    #             print "Accuracy Score: ", results[0]
    #             print
    #             print "Classification Report: "
    #             print results[3]
    #             print
    #             average_accuracy += results[0]
    #             all_results.append(results)

    #             # Save the Results of interest
    #             X_out = []
    #             y_out = []
    #             windows_out = []
    #             prob_out = []
    #             count_out = 0


    #             if True:
    #                 proba = []
    #                 correct_idx = []
    #                 for ii,prob in enumerate(results[1]):
    #                     proba.append(float(prob[1]))
    #                 # Converting windows from np.arrays to lists
    #                 for ii,window_batch in enumerate(window_test):
    #                     for jj,window in enumerate(window_batch):
    #                         window_test[ii][jj] = list(window)


    #                     if y_test[ii] != round(proba[ii]) and count_out<100:
    #                         print "y_test[ii],proba[ii],round(proba[ii])"
    #                         print y_test[ii],proba[ii],round(proba[ii])
    #                         count+=1
    #                         # Missed Window
    #                         X_out.append(X_test[ii])
    #                         y_out.append(y_test[ii])
    #                         windows_out.append(window_test[ii])
    #                         prob_out.append(proba[ii])
    #                     else:
    #                         correct_idx.append(ii)

    #                 # Probably Correct Window
    #                 for ii in range(0,len(prob_out)):
    #                     rand = np.random.randint(0,len(correct_idx)-1)
    #                     X_out.append(X_test[correct_idx[rand]])
    #                     y_out.append(y_test[correct_idx[rand]])
    #                     windows_out.append(window_test[correct_idx[rand]])
    #                     prob_out.append(proba[correct_idx[rand]])
    #                     del correct_idx[rand]

    #             # # Save data in JSON File
    #             # data_out = {}
    #             # data_out['X_out'] = X_out
    #             # data_out['y_out'] = y_out
    #             # data_out['window_out'] = windows_out
    #             # data_out['prob_out'] = prob_out
    #             # data_out['headers'] = headers
    #             # with open('json_data.txt', 'w') as outfile:
    #             #     print "JSON Dumping"
    #             #     json.dump(data_out, outfile)



    #             #Finding Best CV Model
    #             if best_accuracy < results[0]:
    #                 best_accuracy = results[0]
    #                 best_model = ml_model

    #         with open('/mnt/mydata/final/' + 'UW_model_500.pickle', 'wb') as handle:
    #             pickle.dump(ml_model, handle)
    #             print "Model Dumped"


    #         average_accuracy = average_accuracy / num_cv_folds
    #         all_results.append(average_accuracy)
    #         pop_accuracy.append(average_accuracy)
    #         print
    #         print "Average Accuracy Score accross CV: ", average_accuracy
    #         print("\n\n\n" + str(datetime.now()))
    #         all_results.append(datetime.now())


    #         f_name ='/mnt/mydata/final/' + str(num_patients) + '_UW_results_final3.p'

    #         print "Saving Results!"
    #         cPickle.dump(all_results, open( f_name,"wb"))


    # # plot population models
    # f_name = 'UW_population_results_final3.p'
    # cPickle.dump([num_list,pop_accuracy], open( f_name,"wb"))
    # # print plot
    # print "numlist"
    # print num_list
    # print pop_accuracy
    # plt.plot(num_list,pop_accuracy,linewidth=2.0)
    # plt.axis([0,len(num_list)+1,0,1])
    # plt.xlabel('Data Set Size')
    # plt.ylabel('Average Test Set Accuracy (%)')
    # plt.title('UWSleep Population Model Learning Curve')
    # plt.show()




