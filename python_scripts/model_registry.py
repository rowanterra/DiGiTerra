"""Model registry: all train_* functions and get_model_kwargs for DiGiTerra.

Same imports as app_model_training.py (regression, classify, cluster).
MODEL_REGISTRY maps model name -> (train_fn, problem_type).
Polynomial has no train_*; it is registered with train_fn=None.
"""

import ast
from typing import Any, Callable, Dict, Optional, Tuple

# Regression
from python_scripts.models.regression_models.train_linear import train_linear
from python_scripts.models.regression_models.train_lasso import train_lasso
from python_scripts.models.regression_models.train_elasticnet import train_elasticnet
from python_scripts.models.regression_models.train_gb import train_gb
from python_scripts.models.regression_models.train_knn import train_knn
from python_scripts.models.regression_models.train_mlp import train_mlp
from python_scripts.models.regression_models.train_perceptron import train_perceptron
from python_scripts.models.regression_models.train_rf import train_rf
from python_scripts.models.regression_models.train_ridge import train_ridge
from python_scripts.models.regression_models.train_svr import train_svr
from python_scripts.models.regression_models.train_bayesian_ridge import train_bayesian_ridge
from python_scripts.models.regression_models.train_ard_regression import train_ard_regression
from python_scripts.models.regression_models.train_extra_trees import train_extra_trees
from python_scripts.models.regression_models.train_adaboost_regressor import train_adaboost_regressor
from python_scripts.models.regression_models.train_bagging_regressor import train_bagging_regressor
from python_scripts.models.regression_models.train_decision_tree_regressor import train_decision_tree_regressor
from python_scripts.models.regression_models.train_elasticnet_cv import train_elasticnet_cv
from python_scripts.models.regression_models.train_hist_gradient_boosting_regressor import train_hist_gradient_boosting_regressor
from python_scripts.models.regression_models.train_huber_regressor import train_huber_regressor
from python_scripts.models.regression_models.train_lars import train_lars
from python_scripts.models.regression_models.train_lars_cv import train_lars_cv
from python_scripts.models.regression_models.train_lasso_cv import train_lasso_cv
from python_scripts.models.regression_models.train_lassolars import train_lassolars
from python_scripts.models.regression_models.train_linearsvr import train_linearsvr
from python_scripts.models.regression_models.train_nusvr import train_nusvr
from python_scripts.models.regression_models.train_orthogonal_matching_pursuit import train_orthogonal_matching_pursuit
from python_scripts.models.regression_models.train_passive_aggressive_regressor import train_passive_aggressive_regressor
from python_scripts.models.regression_models.train_quantile_regressor import train_quantile_regressor
from python_scripts.models.regression_models.train_radius_neighbors_regressor import train_radius_neighbors_regressor
from python_scripts.models.regression_models.train_ransac_regressor import train_ransac_regressor
from python_scripts.models.regression_models.train_ridge_cv import train_ridge_cv
from python_scripts.models.regression_models.train_sgd_regressor import train_sgd_regressor
from python_scripts.models.regression_models.train_theilsen_regressor import train_theilsen_regressor
# Classify
from python_scripts.models.classify_models.train_logistic_classifier import train_logistic_classifier
from python_scripts.models.classify_models.train_mlp_classifier import train_mlp_classifier
from python_scripts.models.classify_models.train_rf_classifier import train_rf_classifier
from python_scripts.models.classify_models.train_svc import train_svc
from python_scripts.models.classify_models.train_extra_trees_classifier import train_extra_trees_classifier
from python_scripts.models.classify_models.train_gaussian_nb import train_gaussian_nb
from python_scripts.models.classify_models.train_sgd_classifier import train_sgd_classifier
from python_scripts.models.classify_models.train_adaboost_classifier import train_adaboost_classifier
from python_scripts.models.classify_models.train_bagging_classifier import train_bagging_classifier
from python_scripts.models.classify_models.train_bernoulli_nb import train_bernoulli_nb
from python_scripts.models.classify_models.train_categorical_nb import train_categorical_nb
from python_scripts.models.classify_models.train_complement_nb import train_complement_nb
from python_scripts.models.classify_models.train_decision_tree_classifier import train_decision_tree_classifier
from python_scripts.models.classify_models.train_gradient_boosting_classifier import train_gradient_boosting_classifier
from python_scripts.models.classify_models.train_hist_gradient_boosting_classifier import train_hist_gradient_boosting_classifier
from python_scripts.models.classify_models.train_kneighbors_classifier import train_kneighbors_classifier
from python_scripts.models.classify_models.train_linear_discriminant_analysis import train_linear_discriminant_analysis
from python_scripts.models.classify_models.train_linearsvc import train_linearsvc
from python_scripts.models.classify_models.train_multinomial_nb import train_multinomial_nb
from python_scripts.models.classify_models.train_nusvc import train_nusvc
from python_scripts.models.classify_models.train_passive_aggressive_classifier import train_passive_aggressive_classifier
from python_scripts.models.classify_models.train_quadratic_discriminant_analysis import train_quadratic_discriminant_analysis
from python_scripts.models.classify_models.train_ridge_classifier import train_ridge_classifier
# Cluster
from python_scripts.models.cluster_models.train_agglomerative import train_agglomerative
from python_scripts.models.cluster_models.train_gmm import train_gmm
from python_scripts.models.cluster_models.train_kmeans import train_kmeans
from python_scripts.models.cluster_models.train_dbscan import train_dbscan
from python_scripts.models.cluster_models.train_birch import train_birch
from python_scripts.models.cluster_models.train_spectral import train_spectral
from python_scripts.models.cluster_models.train_affinity_propagation import train_affinity_propagation
from python_scripts.models.cluster_models.train_bisecting_kmeans import train_bisecting_kmeans
from python_scripts.models.cluster_models.train_hdbscan import train_hdbscan
from python_scripts.models.cluster_models.train_meanshift import train_meanshift
from python_scripts.models.cluster_models.train_minibatch_kmeans import train_minibatch_kmeans
from python_scripts.models.cluster_models.train_optics import train_optics


def _b(h: dict, k: str, default: bool = True) -> bool:
    """Bool from h: true unless value is 'false'."""
    v = h.get(k)
    if v is None or v == '':
        return default
    return str(v).lower() != 'false'


def _n(h: dict, k: str, default: Any = None) -> Any:
    """Int or 'auto'/'warn' for n_init-style params."""
    v = h.get(k)
    if v is None or v == '':
        return default
    if v in ('auto', 'warn'):
        return v
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _o(h: dict, k: str) -> Optional[Any]:
    """h[k] or None if missing/empty."""
    v = h.get(k)
    if v is None or v == '':
        return None
    return v


MODEL_REGISTRY: Dict[str, Tuple[Optional[Callable], str]] = {
    # Regression
    "Linear": (train_linear, "regression"),
    "BayesianRidge": (train_bayesian_ridge, "regression"),
    "ARDRegression": (train_ard_regression, "regression"),
    "Ridge": (train_ridge, "regression"),
    "Lasso": (train_lasso, "regression"),
    "ElasticNet": (train_elasticnet, "regression"),
    "SVM": (train_svr, "regression"),
    "RF": (train_rf, "regression"),
    "ExtraTrees": (train_extra_trees, "regression"),
    "MLP": (train_mlp, "regression"),
    "K-Nearest": (train_knn, "regression"),
    "gradient_boosting": (train_gb, "regression"),
    "AdaBoost": (train_adaboost_regressor, "regression"),
    "Bagging": (train_bagging_regressor, "regression"),
    "DecisionTree": (train_decision_tree_regressor, "regression"),
    "ElasticNetCV": (train_elasticnet_cv, "regression"),
    "HistGradientBoosting": (train_hist_gradient_boosting_regressor, "regression"),
    "Huber": (train_huber_regressor, "regression"),
    "LARS": (train_lars, "regression"),
    "LARSCV": (train_lars_cv, "regression"),
    "LassoCV": (train_lasso_cv, "regression"),
    "LassoLars": (train_lassolars, "regression"),
    "LinearSVR": (train_linearsvr, "regression"),
    "NuSVR": (train_nusvr, "regression"),
    "OMP": (train_orthogonal_matching_pursuit, "regression"),
    "PassiveAggressive": (train_passive_aggressive_regressor, "regression"),
    "Quantile": (train_quantile_regressor, "regression"),
    "RadiusNeighbors": (train_radius_neighbors_regressor, "regression"),
    "RANSAC": (train_ransac_regressor, "regression"),
    "RidgeCV": (train_ridge_cv, "regression"),
    "SGD": (train_sgd_regressor, "regression"),
    "TheilSen": (train_theilsen_regressor, "regression"),
    "Polynomial": (None, "regression"),  # no train_* in codebase
    "Perceptron": (train_perceptron, "regression"),
    # Classification
    "Logistic_classifier": (train_logistic_classifier, "classification"),
    "ExtraTrees_classifier": (train_extra_trees_classifier, "classification"),
    "GaussianNB_classifier": (train_gaussian_nb, "classification"),
    "SGD_classifier": (train_sgd_classifier, "classification"),
    "MLP_classifier": (train_mlp_classifier, "classification"),
    "RF_classifier": (train_rf_classifier, "classification"),
    "SVC_classifier": (train_svc, "classification"),
    "AdaBoost_classifier": (train_adaboost_classifier, "classification"),
    "Bagging_classifier": (train_bagging_classifier, "classification"),
    "BernoulliNB_classifier": (train_bernoulli_nb, "classification"),
    "CategoricalNB_classifier": (train_categorical_nb, "classification"),
    "ComplementNB_classifier": (train_complement_nb, "classification"),
    "DecisionTree_classifier": (train_decision_tree_classifier, "classification"),
    "GradientBoosting_classifier": (train_gradient_boosting_classifier, "classification"),
    "HistGradientBoosting_classifier": (train_hist_gradient_boosting_classifier, "classification"),
    "KNeighbors_classifier": (train_kneighbors_classifier, "classification"),
    "LDA_classifier": (train_linear_discriminant_analysis, "classification"),
    "LinearSVC_classifier": (train_linearsvc, "classification"),
    "MultinomialNB_classifier": (train_multinomial_nb, "classification"),
    "NuSVC_classifier": (train_nusvc, "classification"),
    "PassiveAggressive_classifier": (train_passive_aggressive_classifier, "classification"),
    "QDA_classifier": (train_quadratic_discriminant_analysis, "classification"),
    "Ridge_classifier": (train_ridge_classifier, "classification"),
    # Cluster
    "kmeans": (train_kmeans, "cluster"),
    "gmm": (train_gmm, "cluster"),
    "agglo": (train_agglomerative, "cluster"),
    "dbscan": (train_dbscan, "cluster"),
    "birch": (train_birch, "cluster"),
    "spectral": (train_spectral, "cluster"),
    "affinity_propagation": (train_affinity_propagation, "cluster"),
    "bisecting_kmeans": (train_bisecting_kmeans, "cluster"),
    "hdbscan": (train_hdbscan, "cluster"),
    "meanshift": (train_meanshift, "cluster"),
    "minibatch_kmeans": (train_minibatch_kmeans, "cluster"),
    "optics": (train_optics, "cluster"),
}

CLUSTER_MODELS: frozenset = frozenset(
    name for name, (_, pt) in MODEL_REGISTRY.items() if pt == "cluster"
)


def _hidden_layer_sizes_tuple(h: Dict[str, Any]) -> str:
    s = '(' + str(h.get('hidden_layer_sizes1', '')) + ',' + str(h.get('hidden_layer_sizes2', ''))
    if h.get('hidden_layer_sizes3'):
        s += ',' + str(h.get('hidden_layer_sizes3', '')) + ')'
    else:
        s += ')'
    return s


def get_model_kwargs(
    modelName: str,
    hyperparameters: Dict[str, Any],
    nonreq: bool,
    seed: Optional[int],
) -> Dict[str, Any]:
    """Return model-specific kwargs (no feature_selection_method/outlier_method/progress_tracker/modeling_mode)."""
    h, n = hyperparameters, nonreq

    if modelName == 'Linear':
        return {}
    if modelName == 'BayesianRidge':
        return {}
    if modelName == 'ARDRegression':
        return {}
    if modelName == 'Ridge':
        if n:
            return {
                'alpha': h.get('alpha'), 'solver': h.get('solver'),
                'RidgeFitIntersept': _b(h, 'RidgeFitIntersept'),
                'RidgeNormalize': _b(h, 'RidgeNormalize'), 'RidgeCopyX': _b(h, 'RidgeCopyX'),
                'RidgePositive': _b(h, 'RidgePositive'), 'RidgeMaxIter': h.get('RidgeMaxIter'),
                'RidgeTol': h.get('RidgeTol'), 'RidgeRandomState': seed,
            }
        return {'alpha': h.get('alpha'), 'RidgeRandomState': seed}
    if modelName == 'Lasso':
        if n:
            return {
                'alpha': h.get('alpha'), 'max_iter': h.get('max_iter'),
                'fit_intercept': _b(h, 'LassoFitIntersept'), 'precompute': _b(h, 'LassoPrecompute'),
                'copy_X': _b(h, 'LassoCopyX'), 'tol': h.get('LassoTol'),
                'warm_start': _b(h, 'LassoWarmStart'), 'positive': _b(h, 'LassoPositive'),
                'random_state': seed, 'selection': h.get('LassoSelection'),
            }
        return {'alpha': h.get('alpha'), 'random_state': seed}
    if modelName == 'ElasticNet':
        return {'alpha': h.get('alpha'), 'l1_ratio': h.get('l1_ratio')}
    if modelName == 'SVM':
        kernel = h.get('kernel', 'rbf')
        base = {'C': h.get('C'), 'kernel': kernel, 'random_state': seed}
        if kernel == 'rbf':
            base['gamma'] = h.get('gamma')
            if n:
                base.update({
                    'coef0': h.get('SVMcoef0'), 'shrinking': _b(h, 'SVMshrinking'),
                    'probability': _b(h, 'SVMprobability'), 'tol': h.get('SVMtol'),
                    'cache_size': h.get('SVMCacheSize'), 'class_weight': h.get('SVMClassWeight'),
                    'verbose': _b(h, 'SVMverbose'), 'max_iter': h.get('SVMmaxIter'),
                    'decision_function_shape': h.get('SVMdecisionFunctionShape'),
                    'break_ties': _b(h, 'SVMBreakTies'),
                })
        elif kernel == 'poly':
            base['degree'] = h.get('degree')
            base['gamma'] = h.get('gamma')
            if n:
                base.update({
                    'coef0': h.get('SVMcoef0'), 'shrinking': _b(h, 'SVMshrinking'),
                    'probability': _b(h, 'SVMprobability'), 'tol': h.get('SVMtol'),
                    'cache_size': h.get('SVMCacheSize'), 'class_weight': h.get('SVMClassWeight'),
                    'verbose': _b(h, 'SVMverbose'), 'max_iter': h.get('SVMmaxIter'),
                    'decision_function_shape': h.get('SVMdecisionFunctionShape'),
                    'break_ties': _b(h, 'SVMBreakTies'),
                })
        elif n:
            base.update({
                'coef0': h.get('SVMcoef0'), 'shrinking': _b(h, 'SVMshrinking'),
                'probability': _b(h, 'SVMprobability'), 'tol': h.get('SVMtol'),
                'cache_size': h.get('SVMCacheSize'), 'class_weight': h.get('SVMClassWeight'),
                'verbose': _b(h, 'SVMverbose'), 'max_iter': h.get('SVMmaxIter'),
                'decision_function_shape': h.get('SVMdecisionFunctionShape'),
                'break_ties': _b(h, 'SVMBreakTies'),
            })
        return base
    if modelName == 'RF':
        val = h.get('max_depth') if 'max_depth' in h else None
        if n:
            return {
                'n_estimators': h.get('n_estimators'), 'max_depth': val,
                'min_samples_split': h.get('min_samples_split'),
                'min_samples_leaf': h.get('min_samples_leaf'), 'random_state': seed,
                'min_weight_fraction_leaf': h.get('RFmin_weight_fraction_leaf'),
                'max_leaf_nodes': h.get('RFMaxLeafNodes'),
                'min_impurity_decrease': h.get('RFMinImpurityDecrease'),
                'bootstrap': _b(h, 'RFBoostrap'), 'oob_score': _b(h, 'RFoobScore'),
                'n_jobs': h.get('RFNJobs'), 'verbose': h.get('RFVerbose'),
                'warm_start': _b(h, 'RFWarmStart'),
            }
        return {'n_estimators': h.get('n_estimators'), 'random_state': seed}
    if modelName == 'ExtraTrees':
        return {}
    if modelName == 'MLP':
        hl = _hidden_layer_sizes_tuple(h)
        if n:
            return {
                'hidden_layer_sizes': ast.literal_eval(hl), 'activation': h.get('activation'),
                'solver': h.get('solver'), 'alpha': h.get('alpha'),
                'learning_rate': h.get('learning_rate'), 'max_iter': h.get('MLPMaxIter'),
                'batch_size': h.get('MLPBatchSize'), 'beta_1': h.get('MLPBeta1'),
                'beta_2': h.get('MLPBeta2'), 'early_stopping': _b(h, 'MLPEarlyStopping'),
                'epsilon': h.get('MLPEpsilon'), 'learning_rate_init': h.get('MLPLearningRateInit'),
                'momentum': h.get('MLPMomentum'), 'nesterovs_momentum': _b(h, 'MLPNesterovsMomentum'),
                'power_t': h.get('MLPPowerT'), 'random_state': seed,
                'shuffle': _b(h, 'MLPShuffle'), 'tol': h.get('MLPTol'),
                'validation_fraction': h.get('MLPValidationFraction'),
                'verbose': _b(h, 'MLPVerbose'), 'warm_start': _b(h, 'MLPWarmStart'),
            }
        return {
            'hidden_layer_sizes': ast.literal_eval(hl), 'activation': h.get('activation'),
            'solver': h.get('solver'), 'random_state': seed,
        }
    if modelName == 'K-Nearest':
        if n:
            return {
                'n_neighbors': h.get('n_neighbors'), 'metric': h.get('metric'),
                'algorithm': h.get('KNearestAlgorithm'), 'leaf_size': h.get('KNearestLeafSize'),
                'metric_params': _o(h, 'KNearestMetricParams'),
                'n_jobs': h.get('KNearestNJobs'), 'p': h.get('KNearestP'),
                'weights': h.get('KNearestWeights'),
            }
        return {'n_neighbors': h.get('n_neighbors')}
    if modelName == 'gradient_boosting':
        if n:
            init = _o(h, 'GBInit')
            max_features = _o(h, 'GBMaxFeatrues')
            return {
                'n_estimators': h.get('n_estimators'), 'learning_rate': h.get('learning_rate'),
                'max_depth': h.get('max_depth'), 'loss': h.get('GBLoss'),
                'subsample': h.get('GBSubsample'), 'criterion': h.get('GBCriterion'),
                'min_samples_split': h.get('GBMinSamplesSplit'),
                'min_samples_leaf': h.get('GBMinSamplesLeaf'),
                'min_weight_fraction_leaf': h.get('GBMinWeightFractionLeaf'),
                'min_impurity_decrease': h.get('GBMinImpurityDecrease'),
                'init': init, 'random_state': seed, 'max_features': max_features,
                'alpha': h.get('GBAlpha'), 'verbose': h.get('GBVerbose'),
                'max_leaf_nodes': h.get('GBMaxLeafNodes'),
                'warm_start': False if h.get('GBWarmStart') else True,
            }
        return {'n_estimators': h.get('n_estimators'), 'learning_rate': h.get('learning_rate'), 'random_state': seed}
    if modelName == 'AdaBoost':
        return {}
    if modelName == 'Bagging':
        return {}
    if modelName == 'DecisionTree':
        return {}
    if modelName == 'ElasticNetCV':
        return {}
    if modelName == 'HistGradientBoosting':
        return {}
    if modelName == 'Huber':
        return {}
    if modelName == 'LARS':
        return {}
    if modelName == 'LARSCV':
        return {}
    if modelName == 'LassoCV':
        return {}
    if modelName == 'LassoLars':
        return {}
    if modelName == 'LinearSVR':
        return {}
    if modelName == 'NuSVR':
        return {}
    if modelName == 'OMP':
        return {}
    if modelName == 'PassiveAggressive':
        return {}
    if modelName == 'Quantile':
        return {}
    if modelName == 'RadiusNeighbors':
        return {}
    if modelName == 'RANSAC':
        return {}
    if modelName == 'RidgeCV':
        return {}
    if modelName == 'SGD':
        return {}
    if modelName == 'TheilSen':
        return {}
    if modelName == 'Polynomial':
        return {'degree': h.get('degree_specificity')}
    if modelName == 'Perceptron':
        if n:
            return {
                'max_iter': h.get('max_iter'), 'eta0': h.get('eta0'),
                'penalty': h.get('PerceptronPenalty'), 'alpha': h.get('PerceptronAlpha'),
                'fit_intercept': _b(h, 'PerceptronFitIntercept'), 'tol': h.get('PerceptronTol'),
                'shuffle': _b(h, 'PerceptronShuffle'), 'verbose': h.get('PerceptronVerbose'),
                'n_jobs': h.get('PerceptronNJobs'), 'random_state': seed,
                'early_stopping': _b(h, 'PerceptronEarlyStopping'),
                'validation_fraction': h.get('PerceptronValidationFraction'),
                'n_iter_no_change': h.get('PerceptronNIterNoChange'),
                'class_weight': h.get('PerceptronClassWeight'), 'warm_start': _b(h, 'PerceptronWarmStart'),
            }
        return {'max_iter': h.get('max_iter'), 'eta0': h.get('eta0')}
    # Classification
    if modelName == 'Logistic_classifier':
        if n:
            return {
                'Class_LogisticDual': _b(h, 'Class_LogisticDual'),
                'Class_LogisticFitIntercept': _b(h, 'Class_LogisticFitIntercept'),
                'Class_LogisticWarmStart': _b(h, 'Class_LogisticWarmStart'),
                'Class_LogisticSolver': h.get('Class_LogisticSolver'),
                'Class_LogisticMultiClass': h.get('Class_LogisticMultiClass'),
                'Class_CLogistic': h.get('Class_CLogistic'),
                'Class_Logistic_penalty': h.get('Class_Logistic_penalty'),
                'Class_LogisticTol': h.get('Class_LogisticTol'),
                'Class_Logisticintercept_scaling': h.get('Class_Logisticintercept_scaling'),
                'Class_LogisticClassWeight': _o(h, 'Class_LogisticClassWeight'),
                'Class_LogisticMaxIterations': h.get('Class_LogisticMaxIterations'),
                'Class_LogisticVerbose': h.get('Class_LogisticVerbose'),
                'Class_LogisticNJobs': _o(h, 'Class_LogisticNJobs'),
                'Class_Logisticl1Ratio': _o(h, 'Class_Logisticl1Ratio'),
            }
        return {}
    if modelName == 'ExtraTrees_classifier':
        return {}
    if modelName == 'GaussianNB_classifier':
        return {}
    if modelName == 'SGD_classifier':
        return {}
    if modelName == 'MLP_classifier':
        hl = _hidden_layer_sizes_tuple(h)
        if n:
            return {
                'hidden_layer_sizes': ast.literal_eval(hl), 'activation': h.get('activation'),
                'solver': h.get('solver'), 'alpha': h.get('alpha'),
                'learning_rate': h.get('learning_rate'), 'max_iter': h.get('MLPMaxIter'),
                'batch_size': h.get('MLPBatchSize'), 'beta_1': h.get('MLPBeta1'),
                'beta_2': h.get('MLPBeta2'), 'early_stopping': _b(h, 'MLPEarlyStopping'),
                'epsilon': h.get('MLPEpsilon'), 'learning_rate_init': h.get('MLPLearningRateInit'),
                'momentum': h.get('MLPMomentum'), 'nesterovs_momentum': _b(h, 'MLPNesterovsMomentum'),
                'power_t': h.get('MLPPowerT'), 'random_state': seed,
                'shuffle': _b(h, 'MLPShuffle'), 'tol': h.get('MLPTol'),
                'validation_fraction': h.get('MLPValidationFraction'),
                'verbose': _b(h, 'MLPVerbose'), 'warm_start': _b(h, 'MLPWarmStart'),
            }
        return {
            'hidden_layer_sizes': ast.literal_eval(hl), 'activation': h.get('activation'),
            'solver': h.get('solver'), 'random_state': seed,
        }
    if modelName == 'RF_classifier':
        val = h.get('max_depth') if 'max_depth' in h else None
        if n:
            return {
                'n_estimators': h.get('n_estimators'), 'max_depth': val,
                'min_samples_split': h.get('min_samples_split'),
                'min_samples_leaf': h.get('min_samples_leaf'), 'random_state': seed,
                'min_weight_fraction_leaf': h.get('RFmin_weight_fraction_leaf'),
                'max_leaf_nodes': h.get('RFMaxLeafNodes'),
                'min_impurity_decrease': h.get('RFMinImpurityDecrease'),
                'bootstrap': _b(h, 'RFBoostrap'), 'oob_score': _b(h, 'RFoobScore'),
                'n_jobs': h.get('RFNJobs'), 'verbose': h.get('RFVerbose'),
                'warm_start': _b(h, 'RFWarmStart'),
            }
        return {'n_estimators': h.get('n_estimators')}
    if modelName == 'SVC_classifier':
        kernel = h.get('kernel', 'rbf')
        base = {'C': h.get('C'), 'kernel': kernel, 'random_state': seed}
        if kernel == 'rbf':
            base['gamma'] = h.get('gamma')
        elif kernel == 'poly':
            base['degree'] = h.get('degree')
            base['gamma'] = h.get('gamma')
        if n:
            base.update({
                'coef0': h.get('SVCcoef0'), 'shrinking': _b(h, 'SVCshrinking'),
                'probability': _b(h, 'SVCprobability'), 'tol': h.get('SVCtol'),
                'cache_size': h.get('SVCCacheSize'), 'class_weight': _o(h, 'SVCClassWeight'),
                'verbose': _b(h, 'SVCverbose'), 'max_iter': h.get('SVCmaxIter'),
                'decision_function_shape': h.get('SVCdecisionFunctionShape'),
                'break_ties': _b(h, 'SVCBreakTies'),
            })
        return base
    if modelName == 'AdaBoost_classifier':
        return {}
    if modelName == 'Bagging_classifier':
        return {}
    if modelName == 'BernoulliNB_classifier':
        return {}
    if modelName == 'CategoricalNB_classifier':
        return {}
    if modelName == 'ComplementNB_classifier':
        return {}
    if modelName == 'DecisionTree_classifier':
        return {}
    if modelName == 'GradientBoosting_classifier':
        return {}
    if modelName == 'HistGradientBoosting_classifier':
        return {}
    if modelName == 'KNeighbors_classifier':
        return {}
    if modelName == 'LDA_classifier':
        return {}
    if modelName == 'LinearSVC_classifier':
        return {}
    if modelName == 'MultinomialNB_classifier':
        return {}
    if modelName == 'NuSVC_classifier':
        return {}
    if modelName == 'PassiveAggressive_classifier':
        return {}
    if modelName == 'QDA_classifier':
        return {}
    if modelName == 'Ridge_classifier':
        return {}
    # Cluster
    if modelName == 'kmeans':
        if n:
            return {
                'n_clusters': h.get('n_clusters'), 'init': h.get('init'),
                'n_init': _n(h, 'n_init'), 'max_iter': h.get('max_iter'),
                'tol': h.get('tol'), 'verbose': h.get('verbose'),
                'copy_x': _b(h, 'copy_x'), 'algorithm': h.get('algorithm'),
            }
        return {'n_clusters': h.get('n_clusters')}
    if modelName == 'gmm':
        if n:
            return {
                'n_components': h.get('n_components'),
                'covariance_type': h.get('covariance_type'),
                'tol': h.get('tol'), 'reg_covar': h.get('reg_covar'),
                'max_iter': h.get('max_iter'), 'n_init': h.get('n_init'),
                'init_params': h.get('init_params'),
                'weights_init': _o(h, 'weights_init'), 'means_init': _o(h, 'means_init'),
                'precisions_init': _o(h, 'precisions_init'),
                'warm_start': _b(h, 'warm_start'),
                'verbose': h.get('verbose'), 'verbose_interval': h.get('verbose_interval'),
            }
        return {'n_components': h.get('n_components')}
    if modelName == 'agglo':
        n_clusters = _o(h, 'n_clusters')
        if n:
            return {
                'n_clusters': n_clusters, 'metric': h.get('metric'),
                'memory': _o(h, 'memory'), 'connectivity': _o(h, 'connectivity'),
                'compute_full_tree': (
                    True if h.get('compute_full_tree') == 'true'
                    else (False if h.get('compute_full_tree') == 'false' else 'auto')
                ),
                'linkage': h.get('linkage'), 'distance_threshold': _o(h, 'distance_threshold'),
                'compute_distances': h.get('distance_threshold') == 'true',
            }
        return {'n_clusters': n_clusters}
    if modelName == 'dbscan':
        return {}
    if modelName == 'birch':
        return {}
    if modelName == 'spectral':
        return {}
    if modelName == 'affinity_propagation':
        return {}
    if modelName == 'bisecting_kmeans':
        return {}
    if modelName == 'hdbscan':
        return {}
    if modelName == 'meanshift':
        return {}
    if modelName == 'minibatch_kmeans':
        return {}
    if modelName == 'optics':
        return {}

    raise KeyError(f"Unknown model: {modelName}")
