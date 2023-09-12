# ClassifyBySklearn
## ClassifyBySklearn.py 
is used for classify things. 
```
usage: ClassifyBySklearn.py [-h] [-t PATH_FEATURE_DF] [-G PATH_GROUP]
                            [-m CLASSIFICATION_METHOD]
                            [-d DECISION_FUNCTION_SHAPE] [-k KERNEL]
                            [-n N_NEIGHBORS] [-f MAX_FEATURES] [-C C] [-T TOL]
                            [-P PENALTY] [-u HIDDEN_UNITS] [-a ACTIVATION_FUN]
                            [-D DROPOUT] [-c EPOCHS] [-b BATCH_SIZE]
                            [-s TRAINING_PART_SCALE] [-g PATH_CHOSEN_GENES]
                            [-v K_CROSS_VALIDATION] [-V V_TIMES_CV]
                            [-r RANDOM_STATE] [-p LOAD_PREDICTOR]
                            [-l LOAD_DICT_DIGIT_TO_GROUP]
                            [-M MINIMUM_VAL_FOR_PREDICTION]
                            [-F MINIMUM_VAL_FOR_DIFFERENCE]
                            [-S SAVE_PREDICTOR] [-y DO_NORMALIZE]
                            [-z DO_NORMALIZE_AFTER_LOC] [-Z OTHER_PARAMETERS]
                            [-o OUTPUT_DIR] [-R TRANSPOSE]
                            [--fillna_genes FILLNA_GENES]

One program for multiple classification method. Notice that filterLowConf by
minimum_val_for_prediction into LOW_CONFIDENCE is only used in the train-test
mode. path_feature_df should be a matrix file with first column as sample
names and first row as feature names. path_group should be a label file with
first column as sample names and second column as labels and doesn't have to
have a header.

optional arguments:
  -h, --help            show this help message and exit
  -t PATH_FEATURE_DF, --path_feature_df PATH_FEATURE_DF
                        Path of the DataFrame (Matrix) for training.
  -G PATH_GROUP, --path_group PATH_GROUP
                        Path of the sample-group table.
  -m CLASSIFICATION_METHOD, --method CLASSIFICATION_METHOD
                        Choices: svm logistic knn rf dt dnn catboost gb nb
  -d DECISION_FUNCTION_SHAPE, --decision_function_shape DECISION_FUNCTION_SHAPE
                        Choices: ovr (1 vs rest) ovo (1 vs 1)
  -k KERNEL, --kernel KERNEL
                        Choices: linear rbf poly sigmoid precomputed
  -n N_NEIGHBORS, --n_neighbors N_NEIGHBORS
                        n_neighbors for knn, or n_estimators for random forest
  -f MAX_FEATURES, --max_features MAX_FEATURES
                        max_features for random forest. Could be int, float,
                        auto, sqrt, log2, None (None means max = all feature
                        num)
  -C C, --C C           For svm or logistic regression
  -T TOL, --tol TOL     For svm or logistic regression
  -P PENALTY, --penalty PENALTY
                        For logistic regression
  -u HIDDEN_UNITS, --hidden_units HIDDEN_UNITS
                        For DNN (space not allowed)
  -a ACTIVATION_FUN, --activation_fun ACTIVATION_FUN
                        for DNN
  -D DROPOUT, --dropout DROPOUT
                        For DNN, not used yet.
  -c EPOCHS, --epochs EPOCHS
                        For SVM/DNN
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        For DNN
  -s TRAINING_PART_SCALE, --training_part_scale TRAINING_PART_SCALE
                        Split scale for train/test. When set to 0, predictor
                        path must be supplied.
  -g PATH_CHOSEN_GENES, --path_chosen_genes PATH_CHOSEN_GENES
                        Path of chosen genes for training or test, if "", use
                        all.
  -v K_CROSS_VALIDATION, --k_cross_validation K_CROSS_VALIDATION
                        k fold cross validation, if None, use the 1-time train
                        test. Else the 1-time train test won't be done.
  -V V_TIMES_CV, --V_times_cv V_TIMES_CV
                        V times of k fold cross validation. Must come together
                        with -v parameter.
  -r RANDOM_STATE, --random_state RANDOM_STATE
                        For 1-time train test or k fold cross validation.
  -p LOAD_PREDICTOR, --load_predictor LOAD_PREDICTOR
                        Load a predictor from load_predictor and no more
                        training. Must come together with
                        load_dict_digit_to_group.
  -l LOAD_DICT_DIGIT_TO_GROUP, --load_dict_digit_to_group LOAD_DICT_DIGIT_TO_GROUP
                        Load the number-group dictionary from
                        load_dict_digit_to_group. Applicable when
                        load_predictor is not empty. (NO MORE NEEDED)
  -M MINIMUM_VAL_FOR_PREDICTION, --minimum_val_for_prediction MINIMUM_VAL_FOR_PREDICTION
                        Predict a class only if its maximum value is above
                        this value. Else classify it as Unknown.)
  -F MINIMUM_VAL_FOR_DIFFERENCE, --minimum_val_for_difference MINIMUM_VAL_FOR_DIFFERENCE
                        Predict a class only if its maximum value is larger
                        than others at this value. Else classify it as
                        Unknown. (Must come together with '-U' parameter. )
  -S SAVE_PREDICTOR, --save_predictor SAVE_PREDICTOR
                        Save predictor instances in the train-test mode.
  -y DO_NORMALIZE, --do_normalize DO_NORMALIZE
                        Normalizer for python. Write in the format that python
                        can eval, such as "Normalizer(norm='l1')" or
                        "QuantileTransformer(output_distribution='normal',
                        n_quantiles=100000)" or "AsIs()" for not doing
                        normalization, "LogAndZeroMeanUnitVar()" for log2
                        transformation followed by standardization
  -z DO_NORMALIZE_AFTER_LOC, --do_normalize_after_loc DO_NORMALIZE_AFTER_LOC
                        Normalizer for python. Write in the format that python
                        can eval, such as "Normalizer(norm='l1')" or
                        "QuantileTransformer(output_distribution='normal',
                        n_quantiles=100000)" or "AsIs()" for not doing
                        normalization, "LogAndZeroMeanUnitVar()" for log2
                        transformation followed by standardization
  -Z OTHER_PARAMETERS, --other_parameters OTHER_PARAMETERS
                        For inputting other parameters not shown above. Must
                        be written as a quoted python dict and '' quoted
                        string. E.g.: -Z '{"optionA":"valueA"}'
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Specify output directory.
  -R TRANSPOSE, --transpose TRANSPOSE
                        Transpose the input feature matrix or not.
  --fillna_genes FILLNA_GENES
                        -1 for not filling

The author of this script is Bo Wang, wangbo@geneis.cn
```

## ChooseMostImportantGenes.py 
is used to choose best features. 
Run ChooseMostImportantFeatures.py -h for usage. 
```
usage: ChooseMostImportantGenes.py [-h] [-t PATH_FEATURE_DF] [-G PATH_GROUP]
                                   [-m CLASSIFICATION_METHOD]
                                   [-d DECISION_FUNCTION_SHAPE] [-k KERNEL]
                                   [-n N_NEIGHBORS] [-f MAX_FEATURES] [-C C]
                                   [-T TOL] [-P PENALTY] [-u HIDDEN_UNITS]
                                   [-a ACTIVATION_FUN] [-D DROPOUT]
                                   [-c EPOCHS] [-b BATCH_SIZE]
                                   [-s TRAINING_PART_SCALE]
                                   [-g PATH_CHOSEN_GENES]
                                   [-v K_CROSS_VALIDATION] [-V V_TIMES_CV]
                                   [-r RANDOM_STATE] [-p LOAD_PREDICTOR]
                                   [-l LOAD_DICT_DIGIT_TO_GROUP]
                                   [-M MINIMUM_VAL_FOR_PREDICTION]
                                   [-F MINIMUM_VAL_FOR_DIFFERENCE]
                                   [-S SAVE_PREDICTOR] [-y DO_NORMALIZE]
                                   [-z DO_NORMALIZE_AFTER_LOC]
                                   [-Z OTHER_PARAMETERS] [-o OUTPUT_DIR]
                                   [-R TRANSPOSE]
                                   [--fillna_genes FILLNA_GENES] [-N GENE_NUM]
                                   [-q PREVIOUS_WEIGHTED_GENE]
                                   [-E FOR_EACH_CLASS]

Choose most important genes by multiple classifiers. Some of the method (dnn,
knn) that are not applicable but being here is because this arg parsing
function is imported from ClassifyBySklearn. Additionally, {'chi2',
'spearman', 'pearson'} are also supported path_feature_df should be a matrix
file with first column as sample names and first row as feature names.
path_group should be a label file with first column as sample names and second
column as labels and doesn't have to have a header.

optional arguments:
  -h, --help            show this help message and exit
  -t PATH_FEATURE_DF, --path_feature_df PATH_FEATURE_DF
                        Path of the DataFrame (Matrix) for training.
  -G PATH_GROUP, --path_group PATH_GROUP
                        Path of the sample-group table.
  -m CLASSIFICATION_METHOD, --method CLASSIFICATION_METHOD
                        Choices: svm logistic knn rf dt dnn catboost gb nb
  -d DECISION_FUNCTION_SHAPE, --decision_function_shape DECISION_FUNCTION_SHAPE
                        Choices: ovr (1 vs rest) ovo (1 vs 1)
  -k KERNEL, --kernel KERNEL
                        Choices: linear rbf poly sigmoid precomputed
  -n N_NEIGHBORS, --n_neighbors N_NEIGHBORS
                        n_neighbors for knn, or n_estimators for random forest
  -f MAX_FEATURES, --max_features MAX_FEATURES
                        max_features for random forest. Could be int, float,
                        auto, sqrt, log2, None (None means max = all feature
                        num)
  -C C, --C C           For svm or logistic regression
  -T TOL, --tol TOL     For svm or logistic regression
  -P PENALTY, --penalty PENALTY
                        For logistic regression
  -u HIDDEN_UNITS, --hidden_units HIDDEN_UNITS
                        For DNN (space not allowed)
  -a ACTIVATION_FUN, --activation_fun ACTIVATION_FUN
                        for DNN
  -D DROPOUT, --dropout DROPOUT
                        For DNN, not used yet.
  -c EPOCHS, --epochs EPOCHS
                        For SVM/DNN
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        For DNN
  -s TRAINING_PART_SCALE, --training_part_scale TRAINING_PART_SCALE
                        Split scale for train/test. When set to 0, predictor
                        path must be supplied.
  -g PATH_CHOSEN_GENES, --path_chosen_genes PATH_CHOSEN_GENES
                        Path of chosen genes for training or test, if "", use
                        all.
  -v K_CROSS_VALIDATION, --k_cross_validation K_CROSS_VALIDATION
                        k fold cross validation, if None, use the 1-time train
                        test. Else the 1-time train test won't be done.
  -V V_TIMES_CV, --V_times_cv V_TIMES_CV
                        V times of k fold cross validation. Must come together
                        with -v parameter.
  -r RANDOM_STATE, --random_state RANDOM_STATE
                        For 1-time train test or k fold cross validation.
  -p LOAD_PREDICTOR, --load_predictor LOAD_PREDICTOR
                        Load a predictor from load_predictor and no more
                        training. Must come together with
                        load_dict_digit_to_group.
  -l LOAD_DICT_DIGIT_TO_GROUP, --load_dict_digit_to_group LOAD_DICT_DIGIT_TO_GROUP
                        Load the number-group dictionary from
                        load_dict_digit_to_group. Applicable when
                        load_predictor is not empty. (NO MORE NEEDED)
  -M MINIMUM_VAL_FOR_PREDICTION, --minimum_val_for_prediction MINIMUM_VAL_FOR_PREDICTION
                        Predict a class only if its maximum value is above
                        this value. Else classify it as Unknown.)
  -F MINIMUM_VAL_FOR_DIFFERENCE, --minimum_val_for_difference MINIMUM_VAL_FOR_DIFFERENCE
                        Predict a class only if its maximum value is larger
                        than others at this value. Else classify it as
                        Unknown. (Must come together with '-U' parameter. )
  -S SAVE_PREDICTOR, --save_predictor SAVE_PREDICTOR
                        Save predictor instances in the train-test mode.
  -y DO_NORMALIZE, --do_normalize DO_NORMALIZE
                        Normalizer for python. Write in the format that python
                        can eval, such as "Normalizer(norm='l1')" or
                        "QuantileTransformer(output_distribution='normal',
                        n_quantiles=100000)" or "AsIs()" for not doing
                        normalization, "LogAndZeroMeanUnitVar()" for log2
                        transformation followed by standardization
  -z DO_NORMALIZE_AFTER_LOC, --do_normalize_after_loc DO_NORMALIZE_AFTER_LOC
                        Normalizer for python. Write in the format that python
                        can eval, such as "Normalizer(norm='l1')" or
                        "QuantileTransformer(output_distribution='normal',
                        n_quantiles=100000)" or "AsIs()" for not doing
                        normalization, "LogAndZeroMeanUnitVar()" for log2
                        transformation followed by standardization
  -Z OTHER_PARAMETERS, --other_parameters OTHER_PARAMETERS
                        For inputting other parameters not shown above. Must
                        be written as a quoted python dict and '' quoted
                        string. E.g.: -Z '{"optionA":"valueA"}'
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Specify output directory.
  -R TRANSPOSE, --transpose TRANSPOSE
                        Transpose the input feature matrix or not.
  --fillna_genes FILLNA_GENES
                        -1 for not filling
  -N GENE_NUM, --gene_num GENE_NUM
                        Gene numbers to choose (For binary-class method, genes
                        generated by different classification instances will
                        be joined)
  -q PREVIOUS_WEIGHTED_GENE, --previous_weighted_gene PREVIOUS_WEIGHTED_GENE
                        Path to a previous .weights_genes file.
  -E FOR_EACH_CLASS, --for_each_class FOR_EACH_CLASS
                        If True, output important_gene_list for each cancer.
                        (Only applicable for SelectKBest function.)

The author of this script is Bo Wang, wangbo@geneis.cn

```

## ClassificationHelperFunctions.py 
contains helper functions for the cross validations or train or tests. 

crossValidation function is used for cross validation. 

trainOnFeatureAAndTestOnFeatureB is used for model training and independent testing.
