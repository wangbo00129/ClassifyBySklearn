#!/data2/wangb/bin/python3
#coding=utf-8
'''
Created on 20190318
Reference: 
SVM: 
https://blog.csdn.net/ericcchen/article/details/79332781
https://blog.csdn.net/qq_38336023/article/details/80762250
KNN: 
https://www.cnblogs.com/xiaotan-code/p/6680438.html

@author: Bo Wang, Yuebin Liang, Siwen Zhang

Output in current dir. 

20190328
    Added cross validation.
        If classifyUsingDFAndGroupTxt(... k_cross_validation=None ... ), do normal train_test. 
        Else do k fold cross validation. 
    Added argparse. 

20190402
    Random state can be fixed by random seed in cross validation.
    Simplify the args. 
        Assume the n_neighbors works as n_estimators in rf. 
        Assume the decision_function_shape works as multi_class in logistic classifier. 
    Added DNN classifier. 

20190403
    Changed DNN classifier to KerasClassifier. 

20190404
    Added dropout arg to cmd line. 
    Modified the default epoch num to 100 for DNN. 
    Modified the prefix style for featurer/model saving. 
    Added max_features arg for rf.  
    Extract arg parsing part as a function for invoke of ChooseMostImportantGenes.py.
    Removed the additional test input. If you want to test on a test set that is not split from training,
        do it manually by setting the training_part_scale to 0. 

20190408
    The report file contains the chosen gene file name. 
    Changed the prefix of the predictor to -- instead of _ between different parameters. 

20190409
    Added the -V parameter for the times of k fold cross validation. 

20190410
    Added the plotRoc function. 

20190412
    Found bug on dnn that cannot run cross validation. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Not fixed!!!!!!!!!!!!!!!!!!!!!!!!
    
    Replace TCGA-xxx to xxx while plotting ROC. 
    
20190416
    Add the random state to path prefix. 
    
20190418
    Adjust the parameter prefix order while outputting. 

20190424
    Added the function to filter bad predictions (two or more similar highest probabilities). 
    -M and -U parameters. (For now, only train-test mode are supported. )
    
20190426
    Added flag -

20190527
    If matrix lacks genes in the gene list, print warning. 
    Yuebin Liang found that the featurer will insert ones at the matrix's first column.
    
20190528
    Changed the normalizer to normalizer and added the parameter. 

20190604
    No more require -t or -G as requiredargs because ChooseMostImportantGenes.py may not need X and Y. 
    Add cross_val_predict result to cross validation result. 

20190605
    Add class_weight as balanced directly to the createNewPredictorInstance function.
    Removed the after testing on the real data. 
    Change accuracy score according to cross_val_predict instead of cross_val_score 
        after which the cross validation accuracy should be 1 number instead of the `fold` numbers. 

20190610
    Change to sklearn.neural_network.MLPClassifier for dnn. 
    Dropout rate are no more applicable. 

20190612 
    Revert: remove class_weight as balanced in the createNewPredictorInstance function.
    Set the max_iter default to -1. 
    Add tol argument. 
    Change gamma to 'scale'. 

20190613
    Add a parameter (dict) for various paramter input. 
    Removing default gamma. If necesary, add gamma in the -Z dict parameter. 
    Add xgboost.
    Add RandomizedSearchCV. 

20190617
    Pull the PredictorClasses outside the createNewPredictorInstance.  

20190618
    Print classification_report more detaily. 

20190619
    Remove ('-U', '--unknown_class_name', dest="unknown_class_name") from argparse. 
    Add decision tree method. 
    Predictor saved as [predictor, dict_digit_to_group]. 
    
20190620
    Fixed plotRoc function. Before the ROC is wrong. 
    Allow non-label input. 
    Fix normalization. 

20190625
    Change do_normalize default to True. 
    Move the normalize process to generateFeaturesAndLabels. 
    Change save_predictor default to True. 

20190627
    Change default hidden units to [100,]

20190711
    You can specify the output path from now on by adding . 


20190718
    Change the formatting output floating point values to 16 for classification_report. 
    
20190719
    Add auc to report. 

20190723
    Change the V time v-fold cross validation to the mean of every cell of each 1 time v-fold cross validation. 

20190731
    Add gradient boost. 

20190807
    Add transpose function for GEO datasets. 
    
20190809
    Remove invalid samples with features not being float. 
    Change output name when neccesary. 

20190810
    Remove duplicate column from the matrix if gene names are duplicate, keeping only the first occurrence. 

20190811
    Add weighted to classifers by default. (At the __main__ part.)

20190813
    Return DataFrame in generateFeaturesAndLabels.  

20190816
    Automatically do exp(x) if a dataframe have values less than 0. 

20191017
    Going to create VotingClassifier when receiving -m params that contains ','. 

20191206
    Change the output format of the predicted result by the existing predictor, i.e. delete the folder name to prevent long prefix name. 

20200110
    Add the normalization method. Changing the do_normalize from True/False to a initiated method object (str), for example, "Normalier(norm='l1')". 

20200117
    Shorten the output prefix by using simpler separation symbols. 

20200213
    Modify the output prefix sep symbols from = and , to + and ..
    Modify the test data demonstration: only output the table for the predictable samples. The LOW_CONFIDENCE samples will not be output. (DELETE_FILTERED as the parameter)  

20200519
    Major modification: add the function for choosing genes while cross validation.

20200527
    Add fit_transform function for AsIs.

'''
# Load libraries
import os
import io
import re
import sys
from pandas import DataFrame as df
from pandas import Series
import pandas as pd
from functools import reduce
from io import StringIO 
# skleran.__version__ is '0.20.3' 

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import *
from sklearn.preprocessing import label_binarize
from numpy import inf
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from scipy.stats import pearsonr, spearmanr

correlation_funcs = {'pearson', 'spearman', 'chi2'}


import numpy as np
import random
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc, accuracy_score, confusion_matrix
from sklearn.externals import joblib 
# from keras.models import Sequential
# from keras.layers import Dense, Dropout 
from xgboost import XGBClassifier

# For ROC
import matplotlib
# font = {'family':'Times New Roman'} 
# matplotlib.rc('font',**font) 
from matplotlib.font_manager import FontManager
from matplotlib.font_manager import FontProperties
import copy 
from sklearn.base import clone

arial_font_path = '/data2/wangb/fonts/msfonts/Arial.TTF'
if os.path.exists(arial_font_path):
    arial_font_properties = FontProperties(fname=arial_font_path)
else:
    arial_font_properties = None


# SEP_BETWEEN_OPTIONS = ','
# SEP_BETWEEN_OPTION_VALUE = '='
SEP_BETWEEN_OPTIONS = '+'
SEP_BETWEEN_OPTION_VALUE = '.'

# If true, the LOW_CONFIDENCE samples will be deleted from the result table. 
DELETE_FILTERED = True



# from matplotlib import font_manager
# find_font_result = FontManager().findfont(FontProperties(family = 'Arial'))
# Reuslt:
# /data2/wangb/anaconda2/envs/tensorflow/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf
# print(find_font_result)
# all_fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# print(all_fonts)
# quit()
from matplotlib import pyplot as plt
from scipy import interp
from itertools import cycle
# from MDNNClassifier import DNN
import argparse
# from sklearn.cross_validation import cross_val_predict

# Global variables. 
KEEP_DECIMAL = 4 
WRITE_TO_DISK_PREFIX_MAX_LENGTH = 240

class AsIs():
    def __init__(self):
        pass
    def fit(self, X):
        pass
    def transform(self, X):
        return X 
    def fit_transform(self, X):
        return X

PredictorClasses = {
    'svm':SVC,
    'logistic': LogisticRegression,
    'knn': KNeighborsClassifier,
    'rf':RandomForestClassifier,
    'dt':DecisionTreeClassifier,
    'dnn':MLPClassifier,
    'xgboost':XGBClassifier,
    'gb':GradientBoostingClassifier,
    }
def createNewPredictorInstance(**kargs):
    method = kargs.get('classification_method', 'svm')
    #  method = 'svm'

    # Predictor is the class
    try:
        Predictor = PredictorClasses[method]
    except:
        raise Exception('Classification method not recognized. ')
    
    
    args_to_init = set(Predictor.__init__.__code__.co_varnames)
    intersect = args_to_init & set(kargs.keys())
    
    paras_to_pass = {i:kargs[i] for i in intersect}    
    print('args to pass:')
    print(paras_to_pass)
    predictor_instance = Predictor(**paras_to_pass)
    
    
    # For prefix, remove build_fn because it's meaningless for users. 
    if 'build_fn' in paras_to_pass:
        paras_to_pass.pop('build_fn')
    prefix = method
    prefix += ''.join(sorted(['{}{}{}{}'.format(SEP_BETWEEN_OPTIONS, k, SEP_BETWEEN_OPTION_VALUE, d) for (k, d) in paras_to_pass.items()]))
    prefix = prefix.replace(' ', '')
    
#     prefix = prefix.replace('[','')
#     prefix = prefix.replace(']','')
#     prefix = prefix.replace(',','_')
    print(Predictor)
    
    return predictor_instance, prefix


def savePickle(element, path):
    with open(path, 'wb') as fw:
        pickle.dump(element, fw)

        
def loadPickle(path):
    with open(path, 'rb') as fr:
        return pickle.load(fr) 

def generateFeaturesAndLabels(path_feature_df, path_group, path_chosen_genes="", return_gene_list=False, transpose=False, \
    do_normalize="Normalizer(norm='l1')",  do_normalize_after_loc="Normalizer(norm='l1')", do_exp_if_negative=False, **kargs):
    '''
    do_normalize: a initiated Normalizer from sklearn.preprocessing. Also it could be "AsIs()", leaving the data as is. This is a normalizer before extracting selected genes.
    do_normalize_after_loc: like the do_normalize, but it's after extracting genes, and do normalize again.  
    '''
    X_df = pd.read_csv(path_feature_df, sep='\t', index_col=0)
    if transpose:
        X_df = X_df.T
    X_df = X_df.fillna(0)

    # Remove the invalid samples. 
    valid_rows = []
    for ind in X_df.index:
        if np.issubdtype(X_df.loc[ind,:], np.number):
            valid_rows.append(ind)
            
    # for i in range(len(sums)):
        # if isinstance(sums[i], float):
            # valid_rows.append(i)
    original_shape = X_df.shape
    X_df = X_df.loc[valid_rows,:]

    if do_exp_if_negative:
        if (X_df < 0).sum().sum() > 0:
            # If any number in the matrix is negative, do exp to whole dataframe. 
            from numpy import exp2
            X_df = exp2(X_df)
    print('X_df from {} filtered to {} due to non-number'.format(original_shape, X_df.shape))
    
    # X_df.to_csv(path_feature_df+'.fillna.xls', sep='\t')
    # print(X_df)
    names = X_df.index
    cols = X_df.columns
    
    groups = readGroupInfo(names, path_group, file_group_header=None) 
    
    if do_normalize:    
        print('Normalizing (shown only one row): \n{}'.format(sum(X_df.iloc[0,:])))
        # normalizer = Normalizer(norm='l1')
        normalizer = eval(do_normalize)
        print('do_normalize is {}'.format(normalizer))
        normalizer.fit(X_df)
        X_tmp = normalizer.transform(X_df)
        X_df.loc[:,:] = X_tmp
        print('Normalized as (shown only one row): \n{}'.format(sum(X_df.iloc[0,:])))


    if path_chosen_genes and os.path.exists(path_chosen_genes):
        print('Using genes from: {}'.format(path_chosen_genes))
        with open(path_chosen_genes) as fr:
            gene_list = [l.strip() for l in fr.readlines()]
            if '' in gene_list:
                gene_list.remove('')
            
        for gene in gene_list:
            if gene not in X_df.columns:
                print('''
                !!!!!!!!!Warning!!!!!!!!!
                {} not in {}
                '''.format(gene, path_feature_df))
                
        gene_list = list(filter(lambda x:x in X_df, gene_list))
        # print('Using genes: {}'.format(gene_list))
        
        X_df = X_df.loc[:, gene_list] 
#         X_df.to_csv('{}_{}genes'.format(path_feature_df, X_df.shape[1]), sep='\t')   
        
        cols_maybe_dup = list(X_df.columns)
        cols_names = []
        col_ids = []
        for col_id in range(len(cols_maybe_dup)):
            if cols_maybe_dup[col_id] not in cols_names:
                col_ids.append(col_id)
                cols_names.append(cols_maybe_dup[col_id])
        X_df = X_df.iloc[:, col_ids]
        # Above is for deduplication. 
        
        
        # print(X_df.columns)
        cols = gene_list     
    
    # !!!!!!!!!!!!!!! modified in 20190527
#     X = X_df.values.tolist()
    
    X = X_df
    print('Read matrix with genes and final shape: {}'.format(X.shape))
    # print('head is \n{}'.format(X_df.head().to_csv(sep='\t'))) 
    if do_normalize_after_loc:    
        print('Normalizing (shown only one row): \n{}'.format(sum(X.iloc[0,:])))
        # normalizer = Normalizer(norm='l1')
        normalizer = eval(do_normalize_after_loc)
        print('do_normalize_after_loc is {}'.format(normalizer))
        normalizer.fit(X)
        X_tmp = normalizer.transform(X)
        X.loc[:,:] = X_tmp
        print('Normalized as (shown only one row): \n{}'.format(sum(X.iloc[0,:])))
    
    if X.shape[0] > 0:
#         print('{} loaded as {}x{} matrix'.format(path_feature_df, len(X), len(X[0])))
        print('{} loaded as {} matrix'.format(path_feature_df, X.shape)) 
    # print('X is \n{}'.format(X))
    
    if return_gene_list:
        return X, groups, cols
    else:
        return X, groups


def classifyUsingDFAndGroupTxt(path_df, path_group, path_chosen_genes="", training_part_scale=0.75, k_cross_validation=None, V_times_cv=None, which_is_1=None,
                               plot=False, random_state=None, load_predictor=None, load_dict_digit_to_group=None, output_dir=".", **kargs):
    
    X, groups = generateFeaturesAndLabels(path_df, path_group, path_chosen_genes, **kargs)
      
    if os.path.exists(load_dict_digit_to_group): 
        dict_digit_to_group = loadPickle(load_dict_digit_to_group)
    else:
        if which_is_1 is None:
            groups, dict_digit_to_group = convertToDigitalLabel(groups, True)
        else:
            groups, dict_digit_to_group = convertToBinaryLabel(groups, which_is_1=which_is_1)
    # print(dict_digit_to_group)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # Train and test. 
    classifyUsingSklearn(X, groups, training_part_scale=training_part_scale, k_cross_validation=k_cross_validation, V_times_cv=V_times_cv, dict_digit_to_group=dict_digit_to_group,
                     plot=plot, random_state=random_state, load_predictor=load_predictor, path_chosen_genes=path_chosen_genes, output_dir=output_dir, **kargs)
    print('done')

def inverseDict(dict_to_reverse):
    dict_reversed = {}
    for k, v in dict_to_reverse.items():
        if v not in dict_reversed:
            dict_reversed[v] = k
        else:
            raise Exception('Duplicate value while inversing dict! ')
    return dict_reversed

def convertToDigitalLabel(groups, sort_only=False):
    set_groups = list(set(groups))
    set_groups.sort()
    digit_group = list(range(len(set_groups)))
    dict_group_digit = dict(zip(set_groups, digit_group))
    dict_digit_group = dict(zip(digit_group, set_groups))
    if sort_only:
        converted_group = groups
    else:
        converted_group = list(map(lambda x:dict_group_digit[x], groups))
    
    return converted_group, dict_digit_group

    
def convertToBinaryLabel(groups, which_is_1):
    converted_groups = [1 if g == which_is_1 else 0 for g in groups]
    dict_digit_to_group = {1:which_is_1, 0: 'not ' + which_is_1}
    return converted_groups, dict_digit_to_group


def writeToFile(string, path, open_method='w'):
    # open_method is 'w' or 'a'
    try:
        with io.open(path, open_method, encoding='utf-8') as fw:
            fw.write(str(string))
    except Exception as e:
        print('writeToFile failed due to {}'.format(e))
        

def calculateSpecificityFromConfusionMatrix(df_confusion_matrix):
    '''
    From: https://stackoverflow.com/questions/27959467/how-to-find-tp-tn-fp-and-fn-values-from-8x8-confusion-matrix
    TP (instance belongs to a, classified as a) = 1086
    FP (instance belongs to others, classified as a) = 7 + 0 + 0 + 1 + 4 + 0 + 0 = 12
    FN (instance belongs to a, classified as others) = 7 + 1 + 0 + 2 + 4 + 0 + 0 = 14
    TN (instance belongs to others, classified as others) = Total instance - (TP + FP + FN)
    Specificity = TN/(TN+FP)
    '''
    cancers = df_confusion_matrix.index
    specificities_per_cancer = Series(index=cancers)
    specificities_to_return = Series(index=cancers)
    sample_nums = Series(index=cancers)
    TN_sum = 0
    FP_sum = 0
    for cancer in cancers:
        sample_nums[cancer] = df_confusion_matrix.loc[cancer, :].sum()  
        TP = df_confusion_matrix.loc[cancer, cancer]
        FP = df_confusion_matrix.loc[:, cancer].sum() - TP
        FP_sum += FP
        FN = df_confusion_matrix.loc[cancer, :].sum() - TP
        TN = df_confusion_matrix.loc[:, :].sum().sum() - FP - TP - FN
        TN_sum += TN
        # print(TN)
        # print(TN+FP)
        # print( TN/(TN+FP))
        specificities_per_cancer[cancer] = TN/(TN+FP)
        
    specificities_to_return = specificities_per_cancer.copy()
    specificities_to_return['micro avg'] = TN_sum/(TN_sum+FP_sum)
    specificities_to_return['macro avg'] = specificities_per_cancer.values.mean()
    specificities_to_return['weighted avg'] = (sample_nums*specificities_per_cancer).sum()/sample_nums.sum()
    print('specificities_to_return is {}'.format(specificities_to_return))
    return specificities_to_return
    

def calculateSpecificity(ground_truth, predictions): 
    tp, tn, fn, fp = 0.0,0.0,0.0,0.0
    for l,m in enumerate(ground_truth):        
        if m==predictions[l] and m==1:
            tp+=1
        if m==predictions[l] and m==0:
            tn+=1
        if m!=predictions[l] and m==1:
            fn+=1
        if m!=predictions[l] and m==0:
            fp+=1
    try:
        spec = tn/(tn+fp)
    except:
        spec = 1
    return spec

def predictionsReport(probas, true_labels, dict_digit_to_group, write_to_disk_prefix=None, \
    write_to_disk_prefix_max_length=WRITE_TO_DISK_PREFIX_MAX_LENGTH, unknown_class_name=None, 
    delete_filtered=DELETE_FILTERED): 
    predicted_digit_label = [list(probas.loc[i,:]).index(probas.loc[i,:].max()) for i in probas.index]
    predicted_str_label = [dict_digit_to_group[d] for d in predicted_digit_label]
    
    df_to_probas_predicted_and_true_label = probas
    df_to_probas_predicted_and_true_label.rename(columns=dict_digit_to_group, inplace=True)
    df_to_probas_predicted_and_true_label.loc[:, 'predicted_label'] = predicted_str_label
    df_to_probas_predicted_and_true_label.loc[:, 'true_label'] = true_labels
    df_to_probas_predicted_and_true_label.loc[:, 'correct'] = df_to_probas_predicted_and_true_label.apply(lambda x: 1 if x['predicted_label'] == x['true_label'] else 0, axis=1)
    
    
    if delete_filtered:
        try:
            print('delete filtered: from \n{} to '.format(df_to_probas_predicted_and_true_label))
        except:
            pass
        df_to_probas_predicted_and_true_label = df_to_probas_predicted_and_true_label.loc[df_to_probas_predicted_and_true_label[~(df_to_probas_predicted_and_true_label[unknown_class_name]==1)].index,:]
        try:print(df_to_probas_predicted_and_true_label)
        except:pass
        true_labels = df_to_probas_predicted_and_true_label.loc[:, 'true_label']
        predicted_str_label = df_to_probas_predicted_and_true_label.loc[:, 'predicted_label'] 
        probas = df_to_probas_predicted_and_true_label.iloc[:,:-3]


    try:
        confusion_mat = confusion_matrix(true_labels, predicted_str_label, 
            labels=[dict_digit_to_group[i] for i in range(len(dict_digit_to_group))])
        df_confusion = df(confusion_mat) 
        df_confusion.rename(index=dict_digit_to_group, columns=dict_digit_to_group, inplace=True)
        str_confusion_report = df_confusion.to_csv(sep='\t')
    except Exception as e:
        try:print('confusion_matrix generation failed due to {}'.format(e))
        except:pass
        str_confusion_report = str(e)
    
    

    if write_to_disk_prefix: 
        if len(os.path.basename(write_to_disk_prefix)) > write_to_disk_prefix_max_length:
            os.path.join(os.path.dirname(write_to_disk_prefix), os.path.basename(write_to_disk_prefix)[len(write_to_disk_prefix)-write_to_disk_prefix_max_length:])
        # if len(write_to_disk_prefix) > write_to_disk_prefix_max_length:
            #write_to_disk_prefix = write_to_disk_prefix[len(write_to_disk_prefix)-write_to_disk_prefix_max_length:]
        # ROC plot. 
        aucs = None
        try: 
            dict_group_to_digit = inverseDict(dict_digit_to_group)
            true_int_labels = [dict_group_to_digit[s] for s in true_labels]
            #auc_macro, auc_micro = plotRoc(true_int_labels, probas, dict_digit_to_group, '{}.ROC.png'.format(write_to_disk_prefix))
            aucs, specs = plotRoc(true_int_labels, probas, dict_digit_to_group, \
                '{}.ROC.png'.format(write_to_disk_prefix),unknown_class_name=unknown_class_name) 
        except Exception as e:
            try:
                print("Plotting err: {}\nROC plot failed. Is there NA in the true labels? ".format(e))
            except:
                pass 
            
        print('printing to file {}'.format(write_to_disk_prefix+'.probas.xls'))
        
        str_probas = df_to_probas_predicted_and_true_label.to_csv(sep='\t') 
        str_classification_report, df_report, accuracy = formated_classification_report(true_labels, predicted_str_label)
        try:
            specificities = calculateSpecificityFromConfusionMatrix(df_confusion)
            df_report['specificity'] = specificities
        except:
            pass
        
        # print(df_report)
        if aucs:
            print(aucs)
            df_report.loc[:, 'auc'] = [aucs.get(i, '') for i in df_report.index] 
            # df_report.loc[:, 'specificity_by_plotRoc'] = [specs.get(i, '') for i in df_report.index]
            df_report.loc['micro avg', 'auc'] = aucs['micro']
            df_report.loc['macro avg', 'auc'] = aucs['macro']
        
        try:
            df_report.loc[:, 'support'] = df_report.loc[:,'support'].astype('int') 
        except:
            print('support astype(int) failed: \n{}'.format(df_report))
        str_classification_report = df_report.to_csv(sep='\t') + '\naccuracy\t{}\n'.format(accuracy)
        # Not real ['accuracy', 'precision'], just for putting the accuracy under the report table. 
        df_report.loc['accuracy', 'precision'] = accuracy
        print('writing to {}*'.format(write_to_disk_prefix))
        writeToFile(str_probas, write_to_disk_prefix+'.probas.xls', 'w') 
        writeToFile(str_classification_report, write_to_disk_prefix+'.report.xls', 'w') 
        writeToFile(str_confusion_report, write_to_disk_prefix+'.confusion.xls', 'w') 
        
    else:
        print('Failed to write to disk') 
        print(str_probas)
        print(str_classification_report)
        print(str_confusion_report)
    
    return accuracy, str_probas, df_report, str_confusion_report
    
    
def formated_classification_report(test_y_list, predicted_str_label, required_fields=['precision','recall','f1-score','support']):
    dict_report = classification_report(test_y_list, predicted_str_label, digits=16, output_dict=True)
    # print(str_report) 
    # str_report = re.sub('  +', '\t', str_report) 
    # str_report = re.sub('\n\t', '\n', str_report)
    # str_report = re.sub('\n+', '\n', str_report)
    df_report = df(dict_report).T.loc[:,required_fields]
    # print('before converting to df: \n{}'.format(str_report))
    # io_tmp = StringIO(str_report.replace('\t',','))   
    # print('middle converting to df: \n{}'.format(str_report.replace('\t',',')))
    # df_report = pd.read_csv(io_tmp, sep=',', index_col=0)
    # print('after converting to df: \n{}'.format(df_report))
    accuracy = accuracy_score(test_y_list, predicted_str_label)
    str_report = df_report.to_csv(sep='\t')
    str_report += '\naccuracy\t{}\n'.format(accuracy)
    #confusion_mat = confusion_matrix(test_y_list, predicted_str_label)
    #str_report += '{}\n'.format(confusion_mat.tolist()).replace('], [', '\n').replace(', ','\t').replace('[','').replace(']','')
    # print('str_report is \n{}'.format(str_report))
    return str_report, df_report, accuracy


# plot_learning_curve  
def plot_learning_curve(estimator, title, X, y, path_output, cv=10, train_sizes=np.linspace(.1, 1.0, 5)): 
    plt.figure()
    plt.title(title)  
    plt.xlabel('Training examples')  
    plt.ylabel('Score')   
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1) 
    train_scores_std = np.std(train_scores, axis=1) 
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid() 

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='g') 
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='r')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='g', label='traning score')  
    plt.plot(train_sizes, test_scores_mean, 'o-', color='r', label='testing score') 
    plt.legend(loc='best')
    plt.savefig(path_output)


def classifyUsingSklearn(X, y_list, training_part_scale=0.75, k_cross_validation=None, V_times_cv=None, dict_digit_to_group=None,
                     featurer=False, polynomial=1, interaction_only=False, plot=False, random_state=None,
                     load_predictor=None, do_normalize=False, path_chosen_genes="", output_dir='.', **kargs): 
    '''
    
    load_predictor: can either be string or predictor instance. 
    featurer: can either be a string or a PolynomialFeatures instance. When not None, overrides polynomial and interaction_only. 
    
    path_chosen_genes here is only for extracting the prefix for the files. 
    
    classification_method='svm', kernel='linear', decision_function_shape='ovr', penalty='l1', C=100, n_neighbors=5, 
    The above args (for constructing predictors) are replaced by **kargs. 
    
    Returns: if load_predictor is not None, return the loaed predictor. Else returns the newly trained predictor.     
    '''
    
    classification_method = kargs['classification_method']
    
    output_prefix_featurer = 'genenum_{}'.format(X.shape[1])
    # savePickle(dict_digit_to_group, '{}.dict_digit_to_group'.format(classification_method))
    dict_group_to_digit =  inverseDict(dict_digit_to_group)
    
    minimum_val_for_prediction = kargs.get('minimum_val_for_prediction', -1)
    minimum_val_for_difference = kargs.get('minimum_val_for_difference', -1)
    unknown_class_name = kargs.get('unknown_class_name', 'LOW_CONFIDENCE')
    
    # Generate polynomial features or not. 
    '''
    Should this be done after split? 
    '''
    # X = np.array(X) 
    
    if featurer:
        if isinstance(featurer, str):
            featurer = joblib.load(featurer)
        else:
            featurer = featurer
        X = featurer.transform(X)
        print('Featurer done and input feature shape is {}'.format(X.shape))
        
    elif featurer is None:
        # Do featurer but not loaded. 
        featurer = PolynomialFeatures(degree=polynomial, interaction_only=interaction_only)
        featurer.fit(X)
        X = featurer.transform(X) 
        joblib.dump(featurer, '{}.featurer'.format(output_prefix_featurer))
    elif featurer is False: 
        pass 
        # Do nothing. 
    # X = X
        
    # Split to train/test dataset. 
    y_list = np.array(y_list)   
    
    if k_cross_validation is not None:
        pass  
    
    elif 0 < training_part_scale < 1: 
        X, test_X, y_list, test_y_list = train_test_split(X, y_list, random_state=random_state, train_size=training_part_scale)
        print('shape after spliting:')
        print(X.shape)
        print(test_X.shape)
        # X and y_list is filtered by indices and for training.
    elif training_part_scale == 0:
        test_X, test_y_list = X, y_list 
    
    # Scale or not. 
    # Standardize features
    # z = (x - u) / s
    # Train model
#     print(X)
#     print(y_list)
    if load_predictor:
        
        # output_prefix_predictor = str(load_predictor).replace('/', '').lstrip('.')
        output_prefix_predictor = str(load_predictor).split('/')[-1]
        output_prefix_predictor = os.path.join(output_dir, output_prefix_predictor)
        
        if isinstance(load_predictor, str):
            # This is the path to the predictor. 
            predictor, dict_digit_to_group = joblib.load(load_predictor)
        else:
            predictor, dict_digit_to_group = load_predictor 
    else:
        # When load_predictor is None. 
        
        # For SVC, add probability as true. 
        kargs['probability'] = True
        
        # For DNN, input and output dim shall be clarified. 
        kargs['input_dim'] = X.shape[1]
        kargs['out_dim'] = len(set(y_list)) 
        predictor, output_prefix_predictor = createNewPredictorInstance(**kargs) 
        if do_normalize:
            output_prefix_predictor = 'Normalized_'+output_prefix_predictor
        # Add the gene list name to the prefix. 
        if os.path.exists(path_chosen_genes):
            output_prefix_predictor += '-g_' + os.path.basename(path_chosen_genes)
        else:
            output_prefix_predictor += '-all_genes'
        
        output_prefix_predictor = '{}Rand{}'.format(output_prefix_predictor, random_state)
        
        output_prefix_predictor = os.path.join(output_dir, output_prefix_predictor)
        
        if k_cross_validation is None:    
            # Do normal training - test. 
            predictor.fit(X, y_list)
            #  w = predictor.coef_[0]
            if kargs['save_predictor']:
                # dump predictor and dict_digit_group to predictor. 
                joblib.dump([predictor, dict_digit_to_group], '{}.predictor'.format(output_prefix_predictor))
        else: 
            
            print('Doing cross validation(s): ')
            vTimesCrossValidation(X, y_list, predictor, output_prefix_predictor, k_cross_validation=k_cross_validation, \
                V_times_cv=V_times_cv, random_state=random_state, dict_group_to_digit=dict_group_to_digit, **kargs)
    
    dict_group_to_digit = inverseDict(dict_digit_to_group)
    # No more necesary. 
#     if classification_method == 'svm' and kernel == 'linear':
#         w = predictor.coef_
#         print('svm trained')
#         savePickle(w, '{}hyperplane.w'.format(output_prefix))
    if k_cross_validation is not None:
        pass    
    elif training_part_scale < 1:
        # Test. 
        # if do_normalize:
            # test_X = normalizer.transform(test_X)
        
        predicted = predictor.predict_proba(test_X)
        predicted = df(predicted)
        predicted.index = test_X.index
        print('type')
        
        print(type(predicted))
        print('len of dict_group_to_digit before filterLowConf is {}'.format(len(dict_group_to_digit)))
        predicted, dict_digit_to_group, dict_group_to_digit = filterLowConf(predicted, minimum_val_for_prediction, minimum_val_for_difference, unknown_class_name, dict_group_to_digit)
        
        print('len of dict_group_to_digit after filterLowConf is {}'.format(len(dict_group_to_digit)))
        # savePickle(test_y_list, '{}.true_label'.format(output_prefix_predictor))
        # savePickle(predicted, '{}.predicted.result'.format(output_prefix_predictor))
        # print('predicted: \n{}'.format(predicted))
        # print('test_y_list: \n{}'.format(test_y_list))
        # print('Converting to int: ')
        # test_y_list_int = [dict_group_to_digit[l] for l in test_y_list]
        # print('Converted: {}'.format(test_y_list_int))
        # Plot ROC. 
        # Need to modify!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         predicted = predicted[:,1]
#         test_y_list = np.array([1,1,1])
        # print(test_y_list_int, predicted, '{}.ROC.png'.format(output_prefix_predictor))        
        # auc_macro, auc_micro = plotRoc(test_y_list_int, predicted, dict_digit_to_group, '{}.ROC.png'.format(output_prefix_predictor))
        
        # print('test_y_list_int: \n{}'.format(test_y_list_int))
        
        # predicted_int_label = np.argmax(predicted, 1)
        # print(predicted_int_label)        
        # predicted_str_label = [dict_digit_to_group[p] for p in predicted_int_label]
        # print('predicted:\n{}'+str(predicted_str_label))
        # writeToFile(str(predicted_str_label), '{}.predicted_label.txt'.format(output_prefix_predictor))
        # writeToFile(str(test_y_list), '{}.true_label.txt'.format(output_prefix_predictor))
        
        # df_to_probas_predicted_and_true_label = df(predicted)
        # df_to_probas_predicted_and_true_label.rename(dict_digit_to_group)
        # df_to_probas_predicted_and_true_label.loc[:, 'predicted_label'] = predicted_str_label
        # df_to_probas_predicted_and_true_label.loc[:, 'true_label'] = test_y_list
        # df_to_probas_predicted_and_true_label.to_csv( '{}.predict.xls'.format(output_prefix_predictor), sep='\t')
        
        # print('true')
        # test_y_list = list(test_y_list)
        # print(test_y_list)
        # str_report, accuracy = formated_classification_report(test_y_list, predicted_str_label)
        #accuracy, str_probas, str_classification_report, str_confusion_report = \
        predictionsReport(predicted, test_y_list, dict_digit_to_group, write_to_disk_prefix=output_prefix_predictor, \
            unknown_class_name=unknown_class_name)

        
    if plot:
        plot2d(X, y_list)
        
    return predictor

def filterLowConf(probas, minimum_val_for_prediction, minimum_val_for_difference, unknown_class_name, dict_group_to_digit):
    '''
    Filter those low conf rows.
    (Set them to 0,0,0,......1,0,0... in which 1 is the column for unknown_class_name. )
    '''
    print('filterLowConf by minimum_val_for_prediction {} minimum_val_for_difference {} to unknown_class_name {}'
          .format(minimum_val_for_prediction, minimum_val_for_difference, unknown_class_name))
    # print("probas before filterLowConf: \n{}".format(str(probas.tolist()).replace('], [', '],\n[')))
    probas = probas.copy()
    if unknown_class_name not in dict_group_to_digit:
        print('{} not in the dict_group_to_digit, appending a new column to probas array. '.format(unknown_class_name))
        # unknown_class_name is not among previous cancer types.         
        dict_group_to_digit[unknown_class_name] = len(dict_group_to_digit)
        # Insert a column as the unknown_class_name probability. 
        #probas = np.concatenate([probas, np.zeros((probas.shape[0], 1))], axis=1)
        probas.loc[:,unknown_class_name] = 0
        '''
        #Before 20190605: 
        return probas
        #Not dealing with if unknown_class_name is absent.  
        '''
    print('dict_group_to_digit changed in filterLowConf to len {}'.format(len(dict_group_to_digit)))
    column_of_unknown_class_name = dict_group_to_digit[unknown_class_name]  
    to_be_set = [0] * probas.shape[1]
    print(probas.shape)
    try:
        to_be_set[column_of_unknown_class_name] = 1
    except:
        to_be_set[-1] = 1
    # Remove those with max below minimum_val_for_prediction. 
    # low_conf_rows = np.max(probas, axis=1) < minimum_val_for_prediction
    low_conf_rows = probas[probas.max(axis=1) < minimum_val_for_prediction].index
    
    print(low_conf_rows)
#     print('filtered rows:')
#     print(low_conf_rows)
    probas.loc[low_conf_rows, :] = to_be_set
    # Remove those with max and 2nd max differs by less than minimum_val_for_difference. 
    for ind in probas.index:
        row_max = probas.loc[ind,:].max()
        thresh = row_max - minimum_val_for_difference
        if (probas.loc[ind,:]>=thresh).sum() >= 2:
        # if sum(probas.values[i] >= (row_max - minimum_val_for_difference)) >=2:
#             print('filtered because minimum_val_for_difference: {}'.format(probas[i]))
            probas.loc[ind,:] = to_be_set
        
    # probas[low_conf_rows] = to_be_set
    dict_digit_to_group = inverseDict(dict_group_to_digit)

    # if delete_filtered:
    #     probas = probas.loc[probas[~(probas[unknown_class_name]==1)].index,:]
    
    return probas, dict_digit_to_group, dict_group_to_digit
    
def plotAxisAndSave(outpath, lw=1, title='ROC for Cancers', plot_legend=False, fontsize=27):


    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    try:
        # fontproperties=arial_font_properties
        plt.xlabel('1 - Specificity', fontproperties=arial_font_properties, fontsize=fontsize)
        plt.ylabel('Sensitivity', fontproperties=arial_font_properties, fontsize=fontsize)
        plt.title(title, fontproperties=arial_font_properties, fontsize=fontsize)
    
    except Exception as e:
        print('Loading local font failed due to {}'.format(e))
        plt.xlabel('1 - Specificity', fontname='Arial', fontsize=fontsize)
        plt.ylabel('Sensitivity', fontname='Arial', fontsize=fontsize)
        plt.title(title, fontname='Arial', fontsize=fontsize)
    
        
    if plot_legend:
        plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(outpath, format=outpath.split('.')[-1])
    print("ROC curve is saved as "+outpath)
    
def plotRoc(y_true, y_predict, group_dict, outpath, roc_size=(9,8), unknown_class_name=None, plot_separately=True, plot_micro_macro=True, plot_legend=False, fontsize=27):
    '''
    Also for specificity calculation. 
    '''
    print('Plotting ROC')
    print('Converting to {} column one-hot matrix'.format(y_predict.shape[1]))
    print('y_true is {}'.format(y_true))
    y_predict = y_predict.values
    # Remove the LOW_CONFIDENCE column. 
    column_num = y_predict.shape[1]
    if unknown_class_name is not None: 
        column_num -= 1 
        y_predict = y_predict[:,:-1]
        
    if column_num <= 2:
        print('binary classification')
        temp = np.zeros(y_predict.shape)
        for i in range(len(y_true)):
            if y_true[i] == 1:
                temp[i, 1] = 1
            else:
                temp[i, 0] = 1
        y_true = temp
    else:
        print('multi classification')
        y_true = label_binarize(y_true, classes=range(column_num))
    print('y_true converted as \n{}'.format(y_true))

    
    ###!!!!!!!!! 
    ### Changed from     
    n_classes = y_true.shape[1]
    print(y_true.shape[1])
    group_dict = group_dict.copy()
    print(group_dict)
    ### To 
#     n_classes = y_predict.shape[1]
    
    print('n_classes: {}'.format(n_classes))
    
    # fpr tpr value
    specs = dict()
    fpr = dict()
    tpr = dict()
    thresh = dict()
    roc_auc = dict()
    for i in range(n_classes):
        print('Generating ROC data for {}'.format(i))
        # print('y_true[:, i]')
        # print(y_true[:, i])
        # print('y_predict[:, i]')
        # print(y_predict[:, i])
        fpr[i], tpr[i], thresh[i] = roc_curve(y_true[:, i], y_predict[:, i])
        # print('fpr[i]')
        # print(fpr[i])
        # print('tpr[i]')
        # print(tpr[i])
        # print('thresh[i]')
        # print(thresh[i])
        roc_auc[group_dict[i]] = auc(fpr[i], tpr[i])
        specs[group_dict[i]] = calculateSpecificity(y_true[:,i], y_predict[:,i])
        # print('roc_auc[i]')
        # print(roc_auc[i])
        
        
    print('specs:\n{}'.format(specs))
    # Compute micro-average ROC curve and ROC area 
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print('roc_auc["micro"] {} '.format(roc_auc["micro"]))
    # Compute macro-average ROC curve and ROC area 
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    # fpr_macro_filtered = fpr['macro']
    # print('fpr["macro"]')
    # print(fpr['macro'])
    # print('tpr["macro"]')
    # print(tpr["macro"])
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print('roc_auc["macro"] {} '.format(roc_auc["macro"]))
    
    # Plot all ROC curves
    lw = 2
    
    
    
    # plt.plot(fpr["micro"], tpr["micro"],
            # label='micro-average (area = {0:0.4f})'.format(roc_auc["micro"]),
            # color='deeppink', linestyle=':', linewidth=4)
    
    # plt.plot(fpr["macro"], tpr["macro"],
        # label='macro-average (area = {0:0.4f})'.format(roc_auc["macro"]),
        # color='navy', linestyle=':', linewidth=4)
        
        
        
    # print('roc_auc["micro"]')
    # print(roc_auc["micro"])
    # print('roc_auc["macro"]')
    # print(roc_auc["macro"])
    

    # Generate random colors with fixed seed. 
    np.random.seed(10)
    # +2 for micro and macro. 
    
    colors_dict = {}
    
    if plot_separately:
        # Use all black. 
        colors_for_n_classes = np.zeros((n_classes+3, 3))
    else:
        colors_for_n_classes = np.random.rand(n_classes+3,3)
    
    group_dict['micro'] = 'micro'
    group_dict['macro'] = 'macro'
    for i in range(len(group_dict)): 
        colors_dict[list(group_dict.keys())[i]] = colors_for_n_classes[i]
    #colors = cycle('bgrcmk')
    # For consistency of each cancer for plotting. 
    
    
    if not plot_separately:
        plt.figure(figsize=roc_size) 
    
    plt.rc('axes', titlesize=fontsize*0.9)     # fontsize of the axes title
    plt.rc('axes', labelsize=fontsize*0.9)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fontsize*0.9)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize*0.9)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=fontsize*0.9)    # legend fontsize
    
    for i in list(range(n_classes))+['micro','macro']:
        if plot_separately:
            plt.figure(figsize=roc_size) 
        print('ploting curve for {}'.format(group_dict[i]))
        # print('fpr[i]')
        # print(fpr[i])
        # print('tpr[i]')
        # print(tpr[i])
        # print('thresh[i]')
        # print(thresh[i])
        # print('roc_auc[i]')
        # print(roc_auc[i])
        try:
            if not plot_micro_macro:
                if i in ['micro','macro']:
                    continue
            plt.plot(fpr[i], tpr[i], color=colors_dict[i], lw=lw, 
                label='{0} (area = {1:0.2f})'.format(group_dict[i].replace('TCGA-',''), roc_auc[group_dict[i]]))
        except Exception as e:
            print('Plotting {}_th class err: {}'.format(i, e))
        if plot_separately:
            outpath_for_i = '.'.join( outpath.split('.')[:-1] + [group_dict[i], outpath.split('.')[-1]] )
            plotAxisAndSave(outpath_for_i, title='ROC for {}'.format(group_dict[i]), fontsize=27)
            
    if not plot_separately:
        plotAxisAndSave(outpath, fontsize=27)
    # Old return. 
    # return roc_auc["macro"], roc_auc["micro"]
    return roc_auc, specs

def plotRocOld(y_true, y_predict, outpath):
    print('Plotting ROC')
#     y_true = np.reshape(y_true, (len(y_true), 1)) 
    print(y_true)
    print('Converting to {} column one-hot matrix'.format(y_predict.shape[1]))
    if y_predict.shape[1]<=2:
        temp = np.zeros(y_predict.shape)
        for i in range(len(y_true)):
            if y_true[i] == 1:
                temp[i, 1] = 1
            else:
                temp[i, 0] = 1
        y_true = temp
    else:
        # This function won't generate one-hot for binary problem. 
        #y_true = label_binarize(y_true, classes=[0,1])
        y_true = label_binarize(y_true, classes=range(y_predict.shape[1]))
    print('y_true converted: \n{}'.format(y_true))
    print(y_predict)
    n_classes = y_true.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area 
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multi-classification')
    plt.legend(loc="lower right")
    plt.savefig(outpath)
    return roc_auc["macro"], roc_auc["micro"] 

def meanAndSdForDataFrames(df_reports):
    mats = []
    result_df = None
    for data_df in df_reports: 
        result_df = data_df
        mats.append(data_df.replace('', 0).values)
        # print(data_df.fillna(''.values)
    mat_3d = np.array(mats)
    print(mat_3d.shape)
    mat_mean = np.mean(mat_3d, axis=0)
    mat_sd = np.std(mat_3d, axis=0)
    df_mean = result_df.copy()
    df_sd = result_df.copy()
    df_mean.loc[:,:] = mat_mean
    df_sd.loc[:,:] = mat_sd
    df_mean.replace(-1, '', inplace=True)
    df_sd.replace(-1, '', inplace=True)
    return df_mean, df_sd


def vTimesCrossValidation(X, y_lsit, predictor, output_prefix_predictor, k_cross_validation, V_times_cv=1, random_state=None, **kargs):
    if V_times_cv is None: 
        V_times_cv = 1
    if V_times_cv == 1:
        # Use random state as the crossValidation seed.
        result = crossValidation(X, y_lsit, predictor, output_prefix_predictor, k_cross_validation, random_state, **kargs)
        return str(round(result, KEEP_DECIMAL))
    else:
        path_cvs_mean_output = '{}.{}Times{}FoldCrossValMean.xls'.format(output_prefix_predictor, V_times_cv, k_cross_validation)
        path_cvs_sd_output = '{}.{}Times{}FoldCrossValSd.xls'.format(output_prefix_predictor, V_times_cv, k_cross_validation)
        print('Performing {} cross validations'.format(V_times_cv))
    
        # Re-generate several numbers for the crossValidation seeds. 
        results = []
        random.seed(random_state)
        random_seeds = random.sample(range(1, 10000), V_times_cv)
        for rs in random_seeds:
#             print('Performing cross validation with random state: {}'.format(rs))
            result = crossValidation(X, y_lsit, predictor, output_prefix_predictor, k_cross_validation, random_state=rs, output_path='{}.{}Times{}CV.Rand{}.txt'.format(output_prefix_predictor, V_times_cv, k_cross_validation, rs), **kargs)
            results.append(result)
        # cvs stands for cross validations. 
        
        # cvs_result = reduce(lambda x,y:'{}\n{}'.format(x,y), results)
        # cvs_mean = np.mean(results)
        # cvs_sd = np.std(results)
        # cvs_result += '\n{} times {}-fold cross validations with initial random state: {}'.format(V_times_cv, k_cross_validation, random_state)
        # cvs_result += '\n' + str(round(cvs_mean, KEEP_DECIMAL)) + '+-' + str(round(cvs_sd, KEEP_DECIMAL))
        df_mean, df_sd = meanAndSdForDataFrames(results)
        df_mean.to_csv(path_cvs_mean_output, sep='\t')
        df_sd.to_csv(path_cvs_sd_output, sep='\t')
        # writeToFile(cvs_result, path_cvs_output, 'w')
        return df_mean, df_sd

def pearson(X, Y):
    result = np.array(list(map(lambda x:pearsonr(x,Y),X.T))).T[0]
    return result

def spearman(X, Y):
    result = np.array(list(map(lambda x:spearmanr(x,Y),X.T))).T[0]
    print(result)
    return result

def extractGeneNameFromBooleanMask(gene_names, bools):
    return np.array(gene_names)[bools].flatten().tolist()

def trainAndChooseGenes(X, y, gene_name_list=None, gene_num=100, dict_digit_group={}, output_prefix='.', **kargs):
    '''
    Only kargs with `feature_selection__` prefix will be taken into acount and all others will be dropped. 
    '''
    kargs = copy.deepcopy(kargs)
    for k in list(kargs.keys()):
        kargs[k.replace('feature_selection__','')] = kargs[k] 
        kargs.pop(k) 
    if gene_num is None:
        print('Not re-selecting genes. ')
        return X.columns 

    print('kargs filtered by feature_selection__:{}'.format(kargs))

    try:
        # Use ClassifyBySklearn.py to select features. 
        predictor, prefix = createNewPredictorInstance(**kargs)
        prefix = output_prefix + prefix
    except:
        # Use feature_selection method like SelectKBest to select features. 
        print(kargs.get('classification_method'))
        try:
            correlation_func = eval(kargs.get('classification_method'))
        except:
            raise Exception('{} are allowed for correlation-based feature selection. '.format(correlation_funcs))

        prefix = output_prefix + kargs.get('classification_method')
        encoder = LabelBinarizer()
        y_onehot= encoder.fit_transform(y)

        all_selecteded_supports = []
        for col in range(y_onehot.shape[1]):
            selector = SelectKBest(correlation_func,k=gene_num).fit(X,y_onehot[:,col])
            if kargs.get('for_each_class'):
                # print(col)
                # print(selector.get_support())
                cancer = dict_digit_group.get(col)
                genes_for_this_cancer = extractGeneNameFromBooleanMask(gene_name_list, selector.get_support())
                writeToFile('\n'.join(genes_for_this_cancer)+'\n', '{}First{}For{}.genes'.format(prefix, gene_num, cancer))
            all_selecteded_supports.append(selector.get_support())
        all_selecteded_genes = reduce(lambda x,y:x+y, all_selecteded_supports)
        print('all_selecteded_genes is :\n{}'.format(all_selecteded_genes))
        # genes_selected = np.array(gene_name_list)[all_selecteded_genes].flatten().tolist()
        genes_selected = extractGeneNameFromBooleanMask(gene_name_list, all_selecteded_genes)
        # print(genes_selected)
        genes_to_write = '\n'.join(genes_selected)+'\n'
        # print(genes_to_write)
        writeToFile(genes_to_write, '{}First{}Union{}.genes'.format(prefix, gene_num, len(genes_selected)))
        return genes_selected
    predictor.fit(X, y)
    try:
        feature_weights = predictor.coef_
    except:
        try:
            feature_weights = predictor.feature_importances_
        except:
            raise Exception('{} method does not own a feature weight list. '.format(type(predictor)))
    print('weight')
    print(feature_weights.shape)
    #     print(feature_weights.sum(axis=1)) 
    savePickle([feature_weights, gene_name_list], prefix+'_feature.weights_genes')
    return chooseGenesByWeights(feature_weights, gene_name_list, gene_num, prefix) 

def chooseGenesByWeights(feature_weights, gene_name_list, gene_num, prefix=None, do_abs=True):
    if do_abs:
        feature_weights = abs(feature_weights)
    indices = np.argsort(feature_weights)
    try:
        # For logistic regression or SVM. 
        reversed_indices = indices[:,::-1]
        first_N_indices = reversed_indices[:, :gene_num]
    except:
        # For random forest. 
        reversed_indices = indices[::-1]
        first_N_indices = reversed_indices[:gene_num]

    if gene_name_list is not None:
        if len(first_N_indices.shape) > 1:
            ind = list(set(first_N_indices.flatten()))
        else:
            ind = list(first_N_indices.flatten())

        ind = np.array(ind)
        gene_list_to_array = np.array(gene_name_list)
        most_important_genes = gene_list_to_array[ind]
        genes_to_write = '\n'.join(most_important_genes)
        if prefix:
            writeToFile(genes_to_write, '{}First{}Union{}.genes'.format(prefix, gene_num, len(most_important_genes)))
    return most_important_genes

def crossValidation(X, y_list, predictor, output_prefix_predictor, k_cross_validation, random_state=None, output_path=None, do_normalize_after_loc=None, **kargs):
    '''
    Not simply a `cross_val_predict` here any more since 20200519. 
    A pipeline containing (1) choose genes (2) evaluate on test part will replace it. 

    do_normalize_after_loc is a str like: "Normalizer(norm='l1')"
    '''
    print('crossValidation {} fold with random seed {}'.format(k_cross_validation, random_state))
    print('X shape: {}'.format(X.shape)) 
    print('y_list len is: {}'.format(len(y_list))) 
    print('Predictor is')
    print(predictor)
    print(type(predictor)) 
    k_fold_stratify_group = StratifiedKFold(n_splits=k_cross_validation, shuffle=True, random_state=random_state) 
    #  cv_score = cross_val_score(predictor, X, y_list, cv=k_fold_stratify_group, n_jobs=3)
#             cv_score = cross_val_score(predictor, X, y_list, cv=k_fold_stratify_group)
#     cv_score = cross_val_score(predictor, X, y_list, cv=k_fold_stratify_group)
    print('cross_val_predict')  


    # This function should be replaced to a custom pipeline. 
    # predictions_after_cross = cross_val_predict(predictor, X, y_list, cv=k_fold_stratify_group, n_jobs=1, method='predict_proba')
    
    gene_num = kargs.get('feature_selection__gene_num')
    predictions_after_cross = []
    for part_id, (train_ind, test_ind) in enumerate(k_fold_stratify_group.split(X, y_list)): 
        print('train ind{}'.format(train_ind))
        print('test ind{}'.format(test_ind))
        X_for_train_in_cv = X.iloc[train_ind,:]
        y_for_train_in_cv = y_list[train_ind] 
        X_for_test_in_cv = X.iloc[test_ind,:]
        y_for_test_in_cv = y_list[test_ind] 
        genes_selected = trainAndChooseGenes(X_for_train_in_cv, y_for_train_in_cv, gene_name_list=list(X_for_train_in_cv.columns), gene_num=gene_num, \
            output_prefix=output_prefix_predictor + 'Rand{}Part{}'.format(random_state,part_id), **kargs) 

        print('genes_selected: \n{}\n{}'.format(genes_selected, len(genes_selected)))
        X_for_train_in_cv = X_for_train_in_cv.loc[:,genes_selected]
        X_for_test_in_cv = X_for_test_in_cv.loc[:,genes_selected]
        predictor_tmp = clone(predictor)


        if do_normalize_after_loc is not None: 
            # Here it's default l1 normalizer. 
            normalizer = eval(do_normalize_after_loc)
            print('do_normalize_after_loc is {}'.format(normalizer))
            X_for_train_in_cv.loc[:,:] = normalizer.fit_transform(X_for_train_in_cv)
            X_for_test_in_cv.loc[:,:] = normalizer.fit_transform(X_for_test_in_cv) 

        predicted_probas_for_current_test_fold = predictor_tmp.fit(X=X_for_train_in_cv,y=y_for_train_in_cv).predict_proba(X_for_test_in_cv)
        
        
        df_predicted_probas_for_current_test_fold = pd.DataFrame(index=X_for_test_in_cv.index, columns=range(predicted_probas_for_current_test_fold.shape[1]))
        df_predicted_probas_for_current_test_fold.loc[:,:] = predicted_probas_for_current_test_fold
        print('df_predicted_probas_for_current_test_fold:\n{}'.format(df_predicted_probas_for_current_test_fold))
        predictions_after_cross.append(df_predicted_probas_for_current_test_fold)
        
    from time import time  
    predictions_after_cross = pd.concat(predictions_after_cross)

    predictions_after_cross = predictions_after_cross.loc[X.index,:]


    predictions_after_cross = df(predictions_after_cross)
    predictions_after_cross.index = X.index
#     print('predictions_after_cross type: {}'.format(type(predictions_after_cross))) 
    print('predictions_after_cross shape: {}'.format(predictions_after_cross.shape))
    dict_group_to_digit = kargs.get('dict_group_to_digit', None).copy()
    
    # predictions_after_cross_with_str_label = predictions_after_cross.copy()
    # dict_digit_to_group = inverseDict(dict_group_to_digit)
    # predictions_after_cross_with_str_label.columns = [dict_digit_to_group[i] for i in predictions_after_cross_with_str_label.columns]
    # predictions_after_cross_with_str_label.loc[:,'true_label'] = y_list
    # predictions_after_cross_with_str_label.to_csv('{}.xls'.format(time()), sep='\t')

    predictions_after_cross, dict_digit_to_group, dict_group_to_digit = filterLowConf(predictions_after_cross, 
                                            kargs.get('minimum_val_for_prediction', -1), 
                                            kargs.get('minimum_val_for_difference', -1), 
                                            kargs.get('unknown_class_name', 'LOW_CONFIDENCE'), 
                                            dict_group_to_digit)
    print('predictions_after_cross after filterLowConf shape: {}'.format(predictions_after_cross.shape))
    # dict_digit_to_group = inverseDict(dict_group_to_digit)
    
    # accuracy_after_cross = sum([z[0]==z[1] for z in zip(y_list, predicted_str_label)])/float(len(y_list))
    # print('accuracy by cross_val_predict')
    # print(accuracy_after_cross)    
    # accuracy_after_cross_by_cross_val_score = cross_val_score(predictor, X, y_list, cv=k_fold_stratify_group)
    # print('accuracy by cross_val_score')
    # print(accuracy_after_cross_by_cross_val_score)
    
    '''
    # Old cross_val_score method outputs `fold` number accuracies in 1 cross validation. 
    cv_score = cross_val_score(predictor, X, y_list, cv=k_fold_stratify_group)
    cv_result = 'random state: {}'.format(random_state)
    cv_result += '\t' + str(cv_score)
    cv_mean = np.mean(cv_score)
    cv_sd = np.std(cv_score)
    cv_result += '\t' + str(round(cv_mean, KEEP_DECIMAL)) + '+-' + str(round(cv_sd, KEEP_DECIMAL))
    cv_result += '\n'
    print(cv_result)
    '''
    
    if output_path is None: 
        path_cv_output = '{}.{}FoldCrossValRand{}.txt'.format(output_prefix_predictor, k_cross_validation, random_state)
    else:
        path_cv_output = output_path
    print('writing to {}'.format(path_cv_output))
    accuracy, str_probas, df_classification_report, str_confusion_report = \
        predictionsReport(predictions_after_cross, y_list, dict_digit_to_group, write_to_disk_prefix=path_cv_output,unknown_class_name=kargs.get('unknown_class_name', 'LOW_CONFIDENCE'))
    
    # writeToFile(cv_result, path_cv_output, 'a')
    # For output
    # true_labels = y_list.tolist()
    # writeToFile('\n'.join([str(p) for p in true_labels])+'\n', path_cv_output+'.cv.true_label', 'w')
    
    # predictions_after_cross = predictions_after_cross.tolist()
    #writeToFile('\n'.join([str(p) for p in predictions_after_cross])+'\n', path_cv_output+'.cv.predict', 'w')
    #writeToFile(str_report+'\naccuracy\t{}'.format(accuracy), path_cv_output+'.report.txt', 'w') 
    # writeToFile(str_probas, path_cv_output+'.probas.txt', 'w') 
    # writeToFile(str_classification_report, path_cv_output+'.report.txt', 'w') 
    # writeToFile(str_confusion_report, path_cv_output+'.confusion.txt', 'w') 
    
    return df_classification_report 


# def plotRoc(labels, predict_prob, output_path):
#     '''
#     From: https://blog.csdn.net/u012875855/article/details/80685221
#     '''
#     false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob)
#     roc_auc = auc(false_positive_rate, true_positive_rate)
#     plt.title('ROC')
#     plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
#     plt.legend(loc='lower right')
#     plt.plot([0, 1], [0, 1], 'r--')
#     plt.ylabel('TPR')
#     plt.xlabel('FPR')
#     plt.savefig(output_path)


# def plot2d(X, y_list):
#     # Plot
#     # color = ['black' if c == 0 else 'red' for c in y_list]
#     color = []
#     for y in y_list:
#         if y == 0:
#             color.append('black')
#         elif y == 1:
#             color.append('red')
#         elif y == 2:
#             color.append('blue')
#         elif y == 3:
#             color.append('green')
#         else:
#             color.append('grey') 
    
#     plt.scatter(X[:, 0], X[:, 1], c=color)
    
#     a = -w[0] / w[1]
#     xx = np.linspace(-2.5, 2.5)
#     yy = a * xx - (predictor.intercept_[0]) / w[1]
        
#     # Plot the hyperplane
#     plt.plot(xx, yy) 
#     #    plt.show()
#     plt.savefig('./a.png')
        
        
def readGroupInfo(names, file_group, file_group_header=None): 
    groups = ['NA'] * len(names)
    if file_group and os.path.exists(file_group): 
        group_df = pd.read_csv(file_group, header=file_group_header, sep='\t', index_col=0)
        # if name in group_df else 'unknown' 
        for i in range(len(names)):
            name = names[i]
            try:
                groups[i] = str(group_df.loc[name, 1])
            except:
                print('''
                !!!!!!!!!!!!!!!!!!!!!!!!
                Warning: {} not in {}
                !!!!!!!!!!!!!!!!!!!!!!!!
                '''.format(name, file_group))
    else:
        print('file for group not exists, going blind. ')
    return groups


def l1(l):
    # Get the first element in the list. For arg parsing. 
    if isinstance(l, list):
        return l[0]
    return l


def mArgParsing(description='this is a description', additional_arg_list=None):
    # Arg parsing method. 
    # additional_arg_list example: gene_num is the dest. 100 is the default. int is the type of this arg. 
    '''
        additional_arg_list = [
                                ['-N', '--gene_num', 'gene_num', 100, 'Gene numbers to choose', int], 
                                ........
                              ]
    ''' 
    
    description += '''
    path_feature_df should be a matrix file with first column as sample names and first row as feature names.  
    path_group should be a label file with first column as sample names and second column as labels and doesn't have to have a header. 
    '''
    parser = argparse.ArgumentParser(description=description, epilog="The author of this script is Bo Wang, wangbo@geneis.cn\n")
#     requiredargs = parser.add_argument_group('required named arguments')
    parser.add_argument('-t', '--path_feature_df', dest="path_feature_df", default=None, help="Path of the DataFrame (Matrix) for training. ", nargs=1, action="store", type=str)
    parser.add_argument('-G', '--path_group', dest="path_group", default=None, help="Path of the sample-group table. ", nargs=1, action="store", type=str)
    parser.add_argument('-m', '--method', dest="classification_method", default='svm', help="Choices: {}".format(' '.join(PredictorClasses.keys())), nargs=1, action="store", type=str)
    parser.add_argument('-d', '--decision_function_shape', dest="decision_function_shape", default='ovr', help="Choices: ovr (1 vs rest) ovo (1 vs 1)", nargs=1, action="store", type=str)
    parser.add_argument('-k', '--kernel', dest="kernel", default='linear', help="Choices: linear rbf poly sigmoid precomputed", nargs=1, action="store", type=str)
    parser.add_argument('-n', '--n_neighbors', dest="n_neighbors", default=5, help="n_neighbors for knn, or n_estimators for random forest", nargs=1, action="store", type=int)
    parser.add_argument('-f', '--max_features', dest="max_features", default='auto', help="max_features for random forest. Could be int, float, auto, sqrt, log2, None (None means max = all feature num)", nargs=1, action="store", type=str)
    parser.add_argument('-C', '--C', dest="C", default=1.0, help="For svm or logistic regression", nargs=1, action="store", type=float)
    parser.add_argument('-T', '--tol', dest="tol", default=0.0001, help="For svm or logistic regression", nargs=1, action="store", type=float)
    parser.add_argument('-P', '--penalty', dest="penalty", default='l1', help="For logistic regression", nargs=1, action="store", type=str)
    parser.add_argument('-u', '--hidden_units', dest="hidden_units", default="[100,]", help="For DNN (space not allowed)", nargs=1, action="store", type=str)
    parser.add_argument('-a', '--activation_fun', dest="activation_fun", default='relu', help="for DNN", nargs=1, action="store", type=str)
    parser.add_argument('-D', '--dropout', dest="dropout", default=0.0, help="For DNN, not used yet. ", nargs=1, action="store", type=float)
    parser.add_argument('-c', '--epochs', dest="epochs", default=-1, help="For SVM/DNN", nargs=1, action="store", type=int)
    parser.add_argument('-b', '--batch_size', dest="batch_size", default=32, help="For DNN", nargs=1, action="store", type=int) 
    parser.add_argument('-s', '--training_part_scale', dest="training_part_scale", default=0.75, help="Split scale for train/test. When set to 0, predictor path must be supplied. ", nargs=1, action="store", type=float)
    parser.add_argument('-g', '--path_chosen_genes', dest="path_chosen_genes", default="", help="Path of chosen genes for training or test, if \"\", use all. ", nargs=1, action="store", type=str)
    parser.add_argument('-v', '--k_cross_validation', dest="k_cross_validation", default=None, help="k fold cross validation, if None, use the 1-time train test. Else the 1-time train test won't be done. ", nargs=1, action="store")
    parser.add_argument('-V', '--V_times_cv', dest="V_times_cv", default=None, help="V times of k fold cross validation. Must come together with -v parameter. ", nargs=1, action="store")
    parser.add_argument('-r', '--random_state', dest="random_state", default=123, help="For 1-time train test or k fold cross validation. ", nargs=1, action="store", type=int)
    parser.add_argument('-p', '--load_predictor', dest="load_predictor", default="", help="Load a predictor from load_predictor and no more training. Must come together with load_dict_digit_to_group. ", nargs=1, action="store", type=str)
    parser.add_argument('-l', '--load_dict_digit_to_group', dest="load_dict_digit_to_group", default="", help="Load the number-group dictionary from load_dict_digit_to_group. Applicable when load_predictor is not empty. (NO MORE NEEDED)", nargs=1, action="store", type=str)
    parser.add_argument('-M', '--minimum_val_for_prediction', dest="minimum_val_for_prediction", default=-1, help="Predict a class only if its maximum value is above this value. Else classify it as Unknown.)", nargs=1, action="store", type=float)
    parser.add_argument('-F', '--minimum_val_for_difference', dest="minimum_val_for_difference", default=-1, help="Predict a class only if its maximum value is larger than others at this value. Else classify it as Unknown. (Must come together with '-U' parameter. )", nargs=1, action="store", type=float)
    # parser.add_argument('-U', '--unknown_class_name', dest="unknown_class_name", default="LOW_CONFIDENCE", help="When the max value is as not large enough, classify to this class. Must be in the previously existing cancer types. ", nargs=1, action="store", type=str)
    parser.add_argument('-S', '--save_predictor', dest="save_predictor", default="True", help="Save predictor instances in the train-test mode. ", nargs=1, action="store", type=str)
    parser.add_argument('-y', '--do_normalize', dest="do_normalize", default="Normalizer(norm='l1')", help="Normalizer for python. Write in the format that python can eval, \
        such as \"Normalizer(norm='l1')\" or \"QuantileTransformer(output_distribution='normal', n_quantiles=100000)\" or \"AsIs()\" for not doing normalization. ", nargs=1, action="store", type=str)
    parser.add_argument('-z', '--do_normalize_after_loc', dest="do_normalize_after_loc", default="Normalizer(norm='l1')", help="Normalizer for python. Write in the format that python can eval, \
        such as \"Normalizer(norm='l1')\" or \"QuantileTransformer(output_distribution='normal', n_quantiles=100000)\" or \"AsIs()\" for not doing normalization. ", nargs=1, action="store", type=str)
    parser.add_argument('-Z', '--other_parameters', dest="other_parameters", default="{}", help="For inputting other parameters not shown above. \n\
                        Must be written as a "" quoted python dict and \'\' quoted string. E.g.: -Z \'{\"optionA\":\"valueA\"}\'", nargs=1, action="store", type=str)
    parser.add_argument('-o', '--output_dir', dest="output_dir", default=".", help="Specify output directory. ", nargs=1, action="store", type=str)
    parser.add_argument('-R', '--transpose', dest="transpose", default="False", help="Transpose the input feature matrix or not. ", nargs=1, action="store", type=str)
    
    
    if additional_arg_list is not None:
        for l in additional_arg_list:
            parser.add_argument(l[0], l[1], dest=l[2], default=l[3], help=l[4], nargs=1, action="store", type=l[5]) 
    
    args = parser.parse_args()
    args.unknown_class_name = 'LOW_CONFIDENCE'
    if args.k_cross_validation is not None:
        args.k_cross_validation = eval(args.k_cross_validation[0])
    if args.V_times_cv is not None:
        args.V_times_cv = eval(args.V_times_cv[0])
        
    try:
        args.max_features = eval(args.max_features[0])
    except:
        pass
    
    if isinstance(l1(args.hidden_units), str):
        args.hidden_units = eval(l1(args.hidden_units))
    print(type(args))
     # n_neighbors and desicion_function_shape works as different args in different methods. 
    args.n_estimators = args.n_neighbors
    args.save_predictor = eval(l1(args.save_predictor))
    args.do_normalize = l1(args.do_normalize)
    args.do_normalize_after_loc = l1(args.do_normalize_after_loc)
    args.multi_class = args.decision_function_shape
    args.other_parameters = eval(l1(args.other_parameters))
    args.transpose = eval(l1(args.transpose))
    print('args.other_parameters: {}'.format(args.other_parameters))
#     print(args.hidden_units)
    return args


if __name__ == '__main__':
    args = mArgParsing(description='''
    One program for multiple classification method.
    Notice that filterLowConf by minimum_val_for_prediction into LOW_CONFIDENCE is only used in the train-test mode. 
    ''') 
    print('args.path_feature_df is \n{}'.format(args.path_feature_df))
    print('args.random_state is {}'.format(args.random_state))
    print('Normalize or not: {}'.format(args.do_normalize)) 
    classifyUsingDFAndGroupTxt(
        path_df=l1(args.path_feature_df),
        path_group=l1(args.path_group),
        training_part_scale=l1(args.training_part_scale),
        path_chosen_genes=l1(args.path_chosen_genes),
        k_cross_validation=l1(args.k_cross_validation),
        V_times_cv=l1(args.V_times_cv),
        classification_method=l1(args.classification_method),
        kernel=l1(args.kernel),
        decision_function_shape=l1(args.decision_function_shape),
        n_neighbors=l1(args.n_neighbors),
        n_estimators=l1(args.n_estimators),
        max_features=l1(args.max_features),
        C=l1(args.C),
        class_weight='balanced',
        penalty=l1(args.penalty), 
        tol=l1(args.tol),  
        # class_weight='balanced', 
        hidden_layer_sizes=args.hidden_units,
        activation=l1(args.activation_fun),
        dropout=l1(args.dropout),
        max_iter=l1(args.epochs),
        batch_size=l1(args.batch_size),
        random_state=l1(args.random_state),
        load_predictor=l1(args.load_predictor),
        load_dict_digit_to_group=l1(args.load_dict_digit_to_group),
        minimum_val_for_prediction=l1(args.minimum_val_for_prediction),
        minimum_val_for_difference=l1(args.minimum_val_for_difference),
        unknown_class_name=l1(args.unknown_class_name),
        save_predictor=l1(args.save_predictor),
        do_normalize=l1(args.do_normalize),
        do_normalize_after_loc=l1(args.do_normalize_after_loc),
        which_is_1=None,
        plot=False, 
        output_dir=l1(args.output_dir),
        transpose=l1(args.transpose),
        **args.other_parameters)
    
#     path_feature_df, path_feature_df_test, path_group, path_chosen_genes, \
#     classification_method, kernel, random_state, load_predictor, load_dict_digit_to_group, k_cross_validation= sys.argv[1:11]
#     
#     random_state = eval(random_state) 
#     k_cross_validation = eval(k_cross_validation)
#     print(k_cross_validation)
#     # which_is_1 = 'Adenocarcinoma, NOS'
#     classifyUsingDFAndGroupTxt(path_feature_df, path_group, path_chosen_genes, k_cross_validation=k_cross_validation, classification_method=classification_method, 
#                                kernel=kernel, 
#                                decision_function_shape='ovr', which_is_1=None, plot=False, random_state=random_state, 
#                                load_predictor=load_predictor, load_dict_digit_to_group=load_dict_digit_to_group, path_feature_df_test=path_feature_df_test)
    
