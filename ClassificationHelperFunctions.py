#!/data2/wangb/bin/python3
# Do cross validation on itself and do train - test on each other. 
# Check the newest version at /data2/wangb/projects/20190221_primary_origin_tracing/home_from_10.0.0.16/classification/IndependentDatasets/CombineAndRun*.py
import os
from os.path import basename, join
from itertools import product
from glob import glob 
import shutil 
import pandas as pd 

def getLabelFile(feature_file, default_label_file_if_missing):
    label_file = feature_file.replace('.TCGA_intersect_STAR_RSEM.genes', '')
    label_file = label_file.replace('.features', '.labels')
    if not os.path.exists(label_file):
        label_file = default_label_file_if_missing     
    return label_file


def sh(cmd):
#     print(cmd)
    if isinstance(cmd, list):
        cmd = ' '.join(map(str, cmd))
    log = 'log'+str(hash(cmd))
    cmd = 'nohup {} > {} 2>&1 '.format(cmd, log)
    print('running cmd:\n{}'.format(cmd))
    os.system(cmd)
    
def crossValidation(feature_file, label_file, feature_selection__classification_method='rf', feature_selection__gene_num=100, params='-m rf -n 2000 -v 10'):
    folder_output = 'cv_{}_Select{}Genes_{}'.format(basename(feature_file), feature_selection__gene_num, params).replace(' ', '_')
    # Escape {} in format string
    # https://stackoverflow.com/questions/19649427/how-can-i-escape-the-format-string
    cmd = ['ClassifyBySklearn.py', '-t', feature_file, '-G', label_file, '-o', folder_output, params, 
           '-Z', '"{{\'feature_selection__classification_method\':\'{}\', \'feature_selection__gene_num\':{}}}"'.format(feature_selection__classification_method, feature_selection__gene_num)]
    cmd = ' '.join(cmd)
    print(cmd)
    # return 
    if os.path.isdir(folder_output):
        print('{} exists, not repeating'.format(folder_output))
    else:
        sh(cmd)

def extractUniqueLabelsForSamples(path_feature, path_label):
    df_features = pd.read_csv(path_feature, sep='\t', index_col=0)
    df_labels = pd.read_csv(path_label, index_col=0, sep='\t', header=None)
    unique_labels = df_labels.loc[df_features.index, 1].unique()
    unique_labels = list(unique_labels)
    return unique_labels

def extractSamplesWithLabels(path_feature, path_label, labels, path_output=None):
    df_features = pd.read_csv(path_feature, sep='\t', index_col=0)
    df_labels = pd.read_csv(path_label, index_col=0, sep='\t', header=None) 
    samples_for_extraction = df_labels[df_labels[1].isin(labels)].index
    df_features = df_features.loc[samples_for_extraction]
    if path_output:
        df_features.to_csv(path_output, sep='\t')
    return df_features

def trainOnFeatureAAndTestOnFeatureB(featureA, labelA, featureB, labelB, gene_file='none', feature_selection__classification_method=None, \
    params_choose_feature='-m rf -n 2000 -s 1.0',
    feature_selection__gene_num=None, params='-m rf -n 2000 -s 1.0', predict_params='-M 0.3 -F 0.1 -s 0.0'):
    folder_output = 'model_{}_{}_{}_Select{}By{}'.format(basename(featureA), basename(gene_file), params, feature_selection__gene_num, feature_selection__classification_method).replace(' ', '_')
    # folder_output_genes = 'genes_{}_{}_{}'.format(basename(featureA), basename(gene_file), params_choose_feature).replace(' ', '_')
    # Choose genes in the training set. 
    if feature_selection__gene_num:
        cmd = ['ChooseMostImportantGenes.py', '-t', featureA, '-G', labelA, '-g', gene_file, params_choose_feature, '-N', feature_selection__gene_num, 
            '-o', folder_output]
        if not glob(folder_output+'/*.genes'):
            sh(cmd) 
        gene_file = glob(folder_output+'/*.genes')[0]
        # Enrichment
        sh('plotGO.R {}'.format(gene_file))
    
    cmd = ['ClassifyBySklearn.py', '-t', featureA, '-G', labelA, '-g', gene_file, '-o', folder_output, params]
    
    if glob(join(folder_output, '*.predictor')):
        print('{} exists, not repeating'.format(folder_output))
    else:
        sh(cmd)

    predictor = glob(join(folder_output, '*.predictor'))[0]
    

    # Test. 
    folder_output = join(folder_output, 'predict_{}_params_{}'.format(basename(featureB), predict_params)).replace(' ','_')
    cmd = ['ClassifyBySklearn.py', '-t', featureB, '-G', labelB, '-g', gene_file, '-o', folder_output, '-p', predictor, '-s', '0.0', predict_params] 
    
    if os.path.isdir(folder_output):
        print('{} exists, not repeating'.format(folder_output))
    else:
        sh(cmd)

def determineWhetherToCVorTest(train_feature, test_feature, gene_num_if_cv=100, gene_file_if_train_test='none', \
    default_label_file_if_missing='/data2/wangb/projects/20190221_primary_origin_tracing/home_from_10.0.0.16/classification/all_rough_group.xls'): 
    '''
    Almost useless since there are too many parameters. 
    '''
    if train_feature == test_feature:
        # Do cross validation. 
        print('cross on {}'.format(train_feature))
        crossValidation(train_feature, getLabelFile(train_feature, default_label_file_if_missing), feature_selection__gene_num=gene_num_if_cv)
    else:
        # Train a model on train_feature and use the predictor to predict the test_feature. 
        print('train test on {} {}'.format(train_feature, test_feature))
        trainOnFeatureAAndTestOnFeatureB(train_feature, getLabelFile(train_feature, default_label_file_if_missing), \
            test_feature, getLabelFile(test_feature, default_label_file_if_missing), 
            gene_file_if_train_test)
