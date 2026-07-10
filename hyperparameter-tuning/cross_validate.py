import os
import yaml
import numpy as np
from copy import deepcopy
from hyperopt import fmin, hp, tpe

import torch

from model import build_model
from train import train, parse_args
from utils import create_logger, get_task_names


# ---------------------------------------
# 10 folds cross validation
# ---------------------------------------
def cross_validate(cfg, logger):
    """k-fold cross-validation.

    """

    # Initialize relevant variables
    init_seed = cfg.SEED
    out_dir = cfg.OUTPUT_DIR
    dataset_mtl_list = ["a_input_plus_random", "b_input_plus_random"] #ZLJ
    task_names=[]
    for dataset_mtl in dataset_mtl_list: #ZLJ
        task_name = get_task_names(os.path.join(cfg.DATA.DATA_PATH, 'raw/{}.csv'.format(dataset_mtl)))
        task_names.append(task_name)
    #task_names = get_task_names(os.path.join(cfg.DATA.DATA_PATH, 'raw/{}.csv'.format(cfg.DATA.DATASET)))

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(cfg.NUM_FOLDS):
        cfg.defrost()
        cfg.SEED = init_seed + fold_num
        cfg.OUTPUT_DIR = os.path.join(out_dir, f'fold_{fold_num}')
        cfg.freeze()
        logger.info(f'Fold {fold_num}')
        model_scores = train(cfg, logger)
        all_scores.append(model_scores)
        #print(all_scores)
    #all_scores = np.array(all_scores) #ZLJ

    # Report results
    cfg.defrost()
    cfg.OUTPUT_DIR = out_dir
    cfg.freeze()
    logger.info(f'{cfg.NUM_FOLDS}-fold cross validation')

    # Report scores for each fold
    """
    for fold_num, scores in enumerate(all_scores):
        logger.info(f'Seed {init_seed + fold_num} ==> test {cfg.DATA.METRIC} = {np.nanmean(scores):.3f}')
        if cfg.SHOW_EACH_SCORES:
            for task_name, score in zip(task_names, scores):
                logger.info(f'Seed {init_seed + fold_num} ==> test {task_name} {cfg.DATA.METRIC} = {score:.3f}')
    """
    a=[]
    b=[]
    for fold_num, scores in enumerate(all_scores): #ZLJ
        print(fold_num, scores)
        for num,i in enumerate(scores):
            logger.info(f'Seed {init_seed + fold_num} ==> test task {num} {cfg.DATA.METRIC} = {np.nanmean(i):.3f}')
            if num==0:
                a.append(np.nanmean(i))
            if num==1:
                b.append(np.nanmean(i))
        #if cfg.SHOW_EACH_SCORES:
            #for task_name, score in zip(task_names, scores):
                #logger.info(f'Seed {init_seed + fold_num} ==> test {task_name} {cfg.DATA.METRIC} = {score:.3f}')
    # Report scores across models
    """
    avg_scores = np.nanmean(all_scores, axis=1)
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    logger.info(f'Overall test {cfg.DATA.METRIC} = {mean_score:.3f} ± {std_score:.3f}')
    """
    
    mean_score_a, std_score_a = np.nanmean(a), np.nanstd(a)
    mean_score_b, std_score_b = np.nanmean(b), np.nanstd(b)
    logger.info(f'Overall test a {cfg.DATA.METRIC} = {mean_score_a:.3f} ± {std_score_a:.3f}')
    logger.info(f'Overall test b {cfg.DATA.METRIC} = {mean_score_b:.3f} ± {std_score_b:.3f}')
    """
    if cfg.SHOW_EACH_SCORES:
        for task_num, task_name in enumerate(task_names):
            logger.info(f'Overall test {task_name} {cfg.DATA.METRIC} = '
                        f'{np.nanmean(all_scores[:, task_num]):.3f} ± {np.nanstd(all_scores[:, task_num]):.3f}')
    """
    
    average = (mean_score_a + mean_score_b) / 2
    return average


# ---------------------------------------
# Hyperparameters optimization
# ---------------------------------------
SPACE = {
    'MODEL.HID': hp.choice('dim', [64, 128, 256]),
    'MODEL.SLICES': hp.choice('slices', [1, 2, 4]),
    'MODEL.DROPOUT': hp.quniform('dropout', low=0.0, high=0.5, q=0.1),
    'MODEL.DEPTH': hp.choice('depth', [2, 3, 4]),
    'TRAIN.OPTIMIZER.BASE_LR': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
    'TRAIN.OPTIMIZER.WEIGHT_DECAY': hp.choice('l2', [1e-4, 1e-5, 1e-6]),
    'DATA.BATCH_SIZE_a': hp.choice('batch_size_a', [8,16, 24,32, 40,48,56,64,72,80,88]), #ZLJ
    'DATA.BATCH_SIZE_b': hp.choice('batch_size_b', [8,16, 24,32, 40,48,56,64]), #ZLJ
}
#INT_KEYS = ['MODEL.HID', 'MODEL.DEPTH', 'MODEL.SLICES']
INT_KEYS = ['MODEL.HID', 'MODEL.DEPTH', 'MODEL.SLICES', 'DATA.BATCH_SIZE_a', 'DATA.BATCH_SIZE_b'] #ZLJ


def hyperopt(cfg, logger):
    """Runs hyperparameter optimization on a HiGNN model.

    """
    # Save path for best hyperparameters
    yaml_name = "best_mtl_{}.yaml".format(cfg.TAG) #ZLJ
    cfg_save_path = os.path.join(cfg.OUTPUT_DIR, yaml_name)
    # Run
    results = []

    # Define hyperparameter optimization
    def objective(hyperparams):
        # Convert hyperparams from float to int when necessary
        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])

        # Update args with hyperparams
        hyper_cfg = deepcopy(cfg)
        if hyper_cfg.OUTPUT_DIR is not None:
            folder_name = f'round_{hyper_cfg.HYPER_COUNT}'
            hyper_cfg.defrost()
            hyper_cfg.OUTPUT_DIR = os.path.join(hyper_cfg.OUTPUT_DIR, folder_name)
            hyper_cfg.freeze()
        hyper_cfg.defrost()
        opts = list()
        for key, value in hyperparams.items():
            opts.append(key)
            opts.append(value)
        hyper_cfg.merge_from_list(opts)
        hyper_cfg.freeze()

        # Record hyperparameters
        cfg.defrost()
        cfg.HYPER_COUNT += 1
        cfg.freeze()
        logger.info(f'round_{hyper_cfg.HYPER_COUNT - 1}')
        logger.info(hyperparams)

        # Cross validate
        #mean_score, std_score = cross_validate(hyper_cfg, logger)
        average = cross_validate(hyper_cfg, logger) #ZLJ
        # Record results
        temp_model = build_model(hyper_cfg)
        num_params = sum(param.numel() for param in temp_model.parameters() if param.requires_grad)
        logger.info(f'num params: {num_params:,}')
        #logger.info(f'{mean_score} ± {std_score} {hyper_cfg.DATA.METRIC}')
        logger.info(f'{average} {hyper_cfg.DATA.METRIC}') #ZLJ
        """
        results.append({
            'mean_score': mean_score,
            'std_score': std_score,
            'hyperparams': hyperparams,
            'num_params': num_params
        })
        """
        results.append({
            'average': average,
            'hyperparams': hyperparams,
            'num_params': num_params
        })
        # Deal with nan
        """
        if np.isnan(mean_score):
            if hyper_cfg.DATA.TASK_TYPE == 'classification':
                mean_score = 0
            else:
                raise ValueError('Can\'t handle nan score for non-classification dataset.')
        """
        if np.isnan(average):
            if hyper_cfg.DATA.TASK_TYPE == 'classification':
                average = 0
            else:
                raise ValueError('Can\'t handle nan score for non-classification dataset.')
        #return (-1 if hyper_cfg.DATA.TASK_TYPE == 'classification' else 1) * mean_score
        return (-1) * average #ZLJ

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=cfg.NUM_ITERS, verbose=False)

    # Report best result
    #results = [result for result in results if not np.isnan(result['mean_score'])]
    results = [result for result in results if not np.isnan(result['average'])]
    best_result = \
        min(results, key=lambda result: (-1) * result['average']) #ZLJ
        #min(results, key=lambda result: (-1 if cfg.DATA.TASK_TYPE == 'classification' else 1) * result['mean_score'])
    logger.info('best result')
    logger.info(best_result['hyperparams'])
    logger.info(f'num params: {best_result["num_params"]:,}')
    #logger.info(f'{best_result["mean_score"]} ± {best_result["std_score"]} {cfg.DATA.METRIC}')
    logger.info(f'{best_result["average"]} {cfg.DATA.METRIC}')
    # Save best hyperparameter settings as yaml config file
    with open(cfg_save_path, 'w') as f:
        yaml.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)


if __name__ == '__main__':
    _, cfg = parse_args()

    logger = create_logger(cfg)

    # print device mode
    if torch.cuda.is_available():
        logger.info('GPU mode...')
    else:
        logger.info('CPU mode...')

    # training
    if cfg.HYPER:
        # Add MODEL.R of the feture attention module
        if cfg.MODEL.F_ATT:
            SPACE.update({'MODEL.R': hp.choice('R', [1, 2, 4])})
            INT_KEYS.append('MODEL.R')
        # Add LOSS.ALPHA and LOSS.TEMPERATURE of the contrastive block
        if cfg.MODEL.BRICS and cfg.LOSS.CL_LOSS:
            SPACE.update({'LOSS.ALPHA': hp.choice('alpha', [0.1, 0.15, 0.2, 0.25])})
            SPACE.update({'LOSS.TEMPERATURE': hp.choice('temperature', [0.07, 0.1, 0.2])})
        # Delete the parameters you don’t want to optimize
        if cfg.HYPER_REMOVE is not None:
            for i in cfg.HYPER_REMOVE:
                del SPACE[i]
            INT_KEYS = [i for i in INT_KEYS if i not in cfg.HYPER_REMOVE]
        hyperopt(cfg, logger)

    else:
        logger.info(cfg.dump())
        cross_validate(cfg, logger)

