#!/data2/wangb/bin/python3
#coding=utf-8
"""
@author: Bo Wang

20190404
    Removed the function createNewPredictorInstance and import it from ClassifyBySklearn.py. 
    Merge the input arg parsing functions with ClassifyBySklearn.py. 
    
20190418
    Output gene importance score (first raw then abs) along with the gene symbol. 

20190627
    Choose only non-zero features. 

20190716
    Add pearsonr as the results. 

20200117
    Shorten the output prefix. 

20200520
    Plans to move trainAndChooseGenes to ClassifyBySklearn.py. (Not done yet)     
"""
import os 
import argparse
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import LabelBinarizer
from functools import reduce

from ClassifyBySklearn import generateFeaturesAndLabels, convertToDigitalLabel, \
    savePickle, loadPickle, writeToFile, createNewPredictorInstance, mArgParsing, l1

def pearson(X, Y):
    result = np.array(list(map(lambda x:pearsonr(x,Y),X.T))).T[0] 
    return result 
    
def spearman(X, Y): 
    result = np.array(list(map(lambda x:spearmanr(x,Y),X.T))).T[0]
    print(result)
    return result 
    
correlation_funcs = {'pearson', 'spearman', 'chi2'}    

def extractGeneNameFromBooleanMask(gene_names, bools):
    return np.array(gene_names)[bools].flatten().tolist()
    

def trainAndChooseGenes(X, y, gene_name_list=None, gene_num=100, dict_digit_group={}, output_dir='./', **kargs):
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        # Use ClassifyBySklearn.py to select features. 
        predictor, prefix = createNewPredictorInstance(**kargs) 
    except:
        # Use feature_selection method like SelectKBest to select features. 
        print(kargs.get('classification_method'))
        try: 
            correlation_func = eval(kargs.get('classification_method')) 
        except:
            raise Exception('{} are allowed for correlation-based feature selection. '.format(correlation_funcs))
            
        prefix = kargs.get('classification_method')
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
                writeToFile('\n'.join(genes_for_this_cancer)+'\n', '{}/{}First{}For{}.genes'.format(output_dir, prefix, gene_num, cancer))
            all_selecteded_supports.append(selector.get_support())
        all_selecteded_genes = reduce(lambda x,y:x+y, all_selecteded_supports)
        print('all_selecteded_genes is :\n{}'.format(all_selecteded_genes))
        # genes_selected = np.array(gene_name_list)[all_selecteded_genes].flatten().tolist()
        genes_selected = extractGeneNameFromBooleanMask(gene_name_list, all_selecteded_genes)
        # print(genes_selected)
        genes_to_write = '\n'.join(genes_selected)+'\n'
        # print(genes_to_write)
        writeToFile(genes_to_write, '{}/{}First{}Union{}.genes'.format(output_dir, prefix, gene_num, len(genes_selected)))
        return 
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
    savePickle([feature_weights, gene_name_list], output_dir+prefix+'_feature.weights_genes')
    chooseGenesByWeights(feature_weights, gene_name_list, gene_num, prefix, output_dir=output_dir)

# Without training, load previous trained weights and gene_names (shall be in one file). 
def loadPreviousAndChooseGenes(previous_weight_and_gene_pickle, gene_num, output_dir='./'):
    os.makedirs(args.output_dir, exist_ok=True)
    feature_weights, gene_name_list = loadPickle(previous_weight_and_gene_pickle)
    chooseGenesByWeights(feature_weights, gene_name_list, gene_num, output_dir=output_dir, prefix=previous_weight_and_gene_pickle.replace('_feature.weights_genes',''))


def chooseGenesByWeights(feature_weights, gene_name_list, gene_num, prefix, output_dir='./', do_abs=True): 
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
        writeToFile(genes_to_write, '{}/{}First{}Union{}.genes'.format(output_dir, prefix, gene_num, len(most_important_genes)))
        
#     savePickle(first_N_indices, 'first_N_indices.indices')
    
    #[::-1]
    
#     indices = np.argsort(importances)[::-1]
#     count2 = 0



if __name__ == '__main__': 
#     argparse.ArgumentParser(description = 'this is a description')
#     parser = argparse.ArgumentParser(description='Parameters',epilog="The author of this script is Bo Wang, wangbo@geneis.cn\n")
#     requiredargs = parser.add_argument_group('required named arguments')
#     requiredargs.add_argument('-t', '--path_feature_df', dest="path_feature_df", default=None, help="Path of the dataframe for training. ", nargs=1,action="store", required=True, type=str)
#     requiredargs.add_argument('-G', '--path_group', dest="path_group", default=None, help="Path of the sample-group table. ", nargs=1,action="store", required=True, type=str)
#     requiredargs.add_argument('-m', '--method', dest="classification_method", default='svm', help="choices: svm knn logistic rf", nargs=1, action="store", type=str)
#     parser.add_argument('-d', '--decision_function_shape', dest="decision_function_shape", default='ovr', help="choices: ovr (1 vs rest) ovo (1 vs 1)", nargs=1, action="store", type=str)
#     parser.add_argument('-k', '--kernel', dest="kernel", default='linear', help="choices: linear, poly, rbf, sigmoid, precomputed, default is linear", nargs=1, action="store", type=str)
#     parser.add_argument('-e', '--n_estimators', dest="n_estimators", default=10, help="n_estimators for RF ", nargs=1, action="store", type=int)
#     parser.add_argument('-n', '--gene_num', dest="gene_num", default=100, help="Gene numbers to choose", nargs=1, action="store", type=int)
    
    additional_arg_list = [['-N', '--gene_num', 'gene_num', 100, 'Gene numbers to choose (For binary-class method, genes generated by different classification instances will be joined)', int], 
                           ['-q', '--previous_weighted_gene', 'previous_weighted_gene', None, 'Path to a previous .weights_genes file. ', str], 
                           ['-E', '--for_each_class', 'for_each_class', "False", 'If True, output important_gene_list for each cancer. (Only applicable for SelectKBest function.)', str], 
                           ]
    args = mArgParsing(description='Choose most important genes by multiple classifiers. \
    Some of the method (dnn, knn) that are not applicable but being here is because this \
    arg parsing function is imported from ClassifyBySklearn. Additionally, {} are also supported'.format(correlation_funcs), additional_arg_list=additional_arg_list)
    
    args.output_dir = l1(args.output_dir)

    # print('using genes from: {}'.format(l1(args.path_chosen_genes))) 
    
    if l1(args.previous_weighted_gene) is None:
        # If previous_weighted_gene not supplied. 
        X, y, gene_name_list = generateFeaturesAndLabels(l1(args.path_feature_df), l1(args.path_group), 
            path_chosen_genes=l1(args.path_chosen_genes), return_gene_list=True, do_normalize=l1(args.do_normalize))
        X = np.array(X) 
        print('X.shape is {}'.format(X.shape))
        y, dict_digit_group = convertToDigitalLabel(y)
        # y is a columns cancer-name-sorted array. 
        
        #     X, test_X, y, test_y_list = train_test_split(X, y, train_size=0.99)
        #     trainAndChooseGenes(X, y, classification_method='rf', kernel='linear', ts='sdf')
        trainAndChooseGenes(X, y, gene_name_list=gene_name_list,                 
                    gene_num=l1(args.gene_num),  
                    classification_method = l1(args.classification_method), 
                    kernel = l1(args.kernel),
                    decision_function_shape = l1(args.decision_function_shape),
                    n_neighbors=l1(args.n_neighbors),
                    n_estimators = l1(args.n_estimators), 
                    max_features = l1(args.max_features),
                    C=l1(args.C),
                    penalty=l1(args.penalty), 
                    do_normalize=l1(args.do_normalize), 
                    for_each_class=eval(l1(args.for_each_class)),
                    dict_digit_group=dict_digit_group,
                    output_dir=args.output_dir, 
                    )
    else:
        loadPreviousAndChooseGenes(l1(args.previous_weighted_gene), gene_num=l1(args.gene_num), output_dir=args.output_dir)    
    
    quit()
    
    
    
        
    # parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0 Beta')
#     parser.add_argument('-e', '--path_feature_df_test', dest="path_feature_df_test", default="", help="path of test data, could be \"\"", nargs=1, action="store", type=str)
#     parser.add_argument('-g', '--path_chosen_genes', dest="path_chosen_genes", default="", help="path of chosen genes for training or test, if \"\", use all. ", nargs=1, action="store", type=str)
#     parser.add_argument('-v', '--k_cross_validation', dest="k_cross_validation", default=None, help="k fold cross validation, if None, use the 1-time train test. Else the 1-time train test won't be done. ", nargs=1, action="store")
#     parser.add_argument('-r', '--random_state', dest="random_state", default=123, help="For 1-time train test. Applicable only when k_cross_validation is None", nargs=1, action="store", type=int)
#     parser.add_argument('-p', '--load_predictor', dest="load_predictor", default="", help="Load a predictor from load_predictor and no more training. ", nargs=1, action="store", type=str)
#     parser.add_argument('-l', '--load_dict_digit_to_group', dest="load_dict_digit_to_group", default=123, help="load the number-group dictionary from load_dict_digit_to_group. Applicable when load_predictor is not empty. ", nargs=1, action="store", type=str)
    
