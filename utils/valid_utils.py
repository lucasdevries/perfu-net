import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
import os, glob
import nibabel as nib
from surface_distance import metrics
import SimpleITK as sitk
plt.rcParams.update({'figure.max_open_warning': 0})

def SaveProbMap(array, wandb_run_dir, name, extension):
    os.makedirs(os.path.join(wandb_run_dir, 'predictions'), exist_ok=True)
    save_path = os.path.join(wandb_run_dir, 'predictions',  name)
    if extension == '.npy':
        with open(save_path, 'wb') as f:
            np.save(f, array.cpu())
    else:
        sitk.WriteImage(sitk.GetImageFromArray(array.cpu()), save_path)

def getScoreDataframe(masks, output, mask_files_complete, metric='dice', threshold=None):
    rows = []
    cases = list(set([msk.split(os.sep)[-1].split('_')[1] for msk in masks]))
    extension = '.' + output[0].split('.')[-1]
    for case in cases:
        slices_mask = sorted([file for file in masks if file.split(os.sep)[-1].split('_')[1] == case])
        slices_output = sorted([file for file in output if file.split(os.sep)[-1].split('_')[1] == case])
        full_mask = [file for file in mask_files_complete if file.split(os.sep)[-1].split('_')[1].split('.')[0] == case]
        full_mask_sitk = sitk.ReadImage(full_mask[0])
        sx, sy, sz = full_mask_sitk.GetSpacing()
        # construct 3d images
        mask_3d = np.zeros((len(slices_mask), 256, 256))
        output_3d = np.zeros((len(slices_mask), 256, 256))
        for ix in range(len(slices_mask)):
            mask_file = slices_mask[ix]
            out_file = slices_output[ix]
            assert (mask_file.split(os.sep)[-1].split('_')[-1] == out_file.split(os.sep)[-1].split('_')[-1])
            if extension == '.npy':
                with open(mask_file, 'rb') as f:
                    mask_3d[ix] = np.load(f)
                with open(out_file, 'rb') as f:
                    output_3d[ix] = np.load(f)
            else:
                mask_3d[ix] = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))
                output_3d[ix] = sitk.GetArrayFromImage(sitk.ReadImage(out_file))
        # Get scores
        scores = {'case': case, 'area': np.sum(mask_3d)}

        if threshold is not None:
            segmentation = (output_3d > threshold)
            if metric == 'dice':
                score = np.sum(segmentation[mask_3d == 1]) * 2.0 / (np.sum(segmentation) + np.sum(mask_3d))

            elif metric == 'recall':
                score = np.sum(segmentation[mask_3d == 1]) / np.sum(mask_3d)

            elif metric == 'precision':
                if np.sum(segmentation) == 0:
                    score = np.nan
                else:
                    score = np.sum(segmentation[mask_3d == 1]) / np.sum(segmentation)

            elif metric == 'volume':
                score = ((np.sum(segmentation) - np.sum(mask_3d)) * sx * sy * sz) / 1000
            elif metric == 'abs_volume':
                score = np.abs(((np.sum(segmentation) - np.sum(mask_3d)) * sx * sy * sz) / 1000)
            elif metric == 'surface_dice':
                distances = metrics.compute_surface_distances(mask_3d.astype(np.bool), segmentation.astype(np.bool), [sz, sx, sy])
                score = metrics.compute_surface_dice_at_tolerance(distances, tolerance_mm=2)
            elif metric == 'hd95':
                distances = metrics.compute_surface_distances(mask_3d.astype(np.bool), segmentation.astype(np.bool), [sz, sx, sy])
                score = metrics.compute_robust_hausdorff(distances, percent=95)
            elif metric == 'hd100':
                distances = metrics.compute_surface_distances(mask_3d.astype(np.bool), segmentation.astype(np.bool), [sz, sx, sy])
                score = metrics.compute_robust_hausdorff(distances, percent=100)
            else:
                raise Exception('Please provide correct metric')
            scores[threshold] = score
        else:
            for i in range(1, 100):
                i = i / 100
                segmentation = (output_3d > i)
                if metric == 'dice':
                    score = np.sum(segmentation[mask_3d == 1]) * 2.0 / (np.sum(segmentation) + np.sum(mask_3d))

                elif metric == 'recall':
                    score = np.sum(segmentation[mask_3d == 1]) / np.sum(mask_3d)

                elif metric == 'precision':
                    if np.sum(segmentation) == 0:
                        score = np.nan
                    else:
                        score = np.sum(segmentation[mask_3d == 1]) / np.sum(segmentation)

                elif metric == 'volume':
                    score = ((np.sum(segmentation) - np.sum(mask_3d)) * sx * sy * sz) / 1000
                elif metric == 'abs_volume':
                    score = np.abs(((np.sum(segmentation) - np.sum(mask_3d)) * sx * sy * sz) / 1000)
                elif metric == 'surface_dice':
                    distances = metrics.compute_surface_distances(mask_3d.astype(np.bool), segmentation.astype(np.bool), [sz, sx, sy])
                    score = metrics.compute_surface_dice_at_tolerance(distances, tolerance_mm=2)
                elif metric == 'hd95':
                    distances = metrics.compute_surface_distances(mask_3d.astype(np.bool), segmentation.astype(np.bool), [sz, sx, sy])
                    score = metrics.compute_robust_hausdorff(distances, percent=95)
                elif metric == 'hd100':
                    distances = metrics.compute_surface_distances(mask_3d.astype(np.bool), segmentation.astype(np.bool), [sz, sx, sy])
                    score = metrics.compute_robust_hausdorff(distances, percent=100)
                else:
                    raise Exception('Please provide correct metric')

                scores[i] = score
        rows.append(scores)
    df = pd.DataFrame.from_dict(rows, orient='columns')
    return df


def getMetrics(config, wandb_run_dir, epoch, best_up_to_now=0):
    # Load sweeps outputs for val. set
    sweep_out_dir = os.path.join(wandb_run_dir, 'predictions{}*{}'.format(os.sep, config.file_extension))
    sweep_out_files = glob.glob(sweep_out_dir)
    mask_files = getValData(config, validation=True)
    mask_files_complete = getValData(config, validation=True, complete=True)
    assert (len(mask_files) == len(sweep_out_files))
    # generate dataframes

    dice_df = getScoreDataframe(mask_files, sweep_out_files, mask_files_complete, metric='dice', threshold=0.5)
    prec_df = getScoreDataframe(mask_files, sweep_out_files, mask_files_complete, metric='precision', threshold=0.5)
    rec_df = getScoreDataframe(mask_files, sweep_out_files, mask_files_complete, metric='recall', threshold=0.5)
    vol_df = getScoreDataframe(mask_files, sweep_out_files, mask_files_complete, metric='volume', threshold=0.5)
    surface_dice_df = getScoreDataframe(mask_files, sweep_out_files, mask_files_complete, metric='surface_dice', threshold=0.5)
    hd95_df = getScoreDataframe(mask_files, sweep_out_files, mask_files_complete, metric='hd95', threshold=0.5)
    hd100_df = getScoreDataframe(mask_files, sweep_out_files, mask_files_complete, metric='hd100', threshold=0.5)
    abs_vol_df = getScoreDataframe(mask_files, sweep_out_files, mask_files_complete, metric='abs_volume', threshold=0.5)

    hd95_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    hd100_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    surface_dice_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    surface_dice_at_top_dice = surface_dice_df[0.5].mean()
    hd95_at_top_dice = hd95_df[0.5].mean()
    hd100_at_top_dice = hd100_df[0.5].mean()

    precision_at_top_dice = prec_df[0.5].mean()
    recall_at_top_dice = rec_df[0.5].mean()
    volume_at_top_dice = vol_df[0.5].mean()
    abs_volume_at_top_dice = abs_vol_df[0.5].mean()

    df2 = dice_df.drop(columns='area')
    top_dice_per_case = df2.groupby('case')[0.5].mean()
    top_dice_mean = top_dice_per_case.mean()

    if top_dice_mean >= best_up_to_now:
        # save dataframes
        os.makedirs(os.path.join(wandb_run_dir, 'pickles'), exist_ok=True)
        dice_df.to_pickle(os.path.join(wandb_run_dir, 'pickles', 'dice-{}'.format(epoch)))
        prec_df.to_pickle(os.path.join(wandb_run_dir, 'pickles', 'precision-{}'.format(epoch)))
        rec_df.to_pickle(os.path.join(wandb_run_dir, 'pickles', 'recall-{}'.format(epoch)))
        vol_df.to_pickle(os.path.join(wandb_run_dir, 'pickles', 'volume-{}'.format(epoch)))
        surface_dice_df.to_pickle(os.path.join(wandb_run_dir, 'pickles', 'surface_dice-{}'.format(epoch)))
        hd95_df.to_pickle(os.path.join(wandb_run_dir, 'pickles', 'hd95-{}'.format(epoch)))
        hd100_df.to_pickle(os.path.join(wandb_run_dir, 'pickles', 'hd100-{}'.format(epoch)))
        abs_vol_df.to_pickle(os.path.join(wandb_run_dir, 'pickles', 'absvol-{}'.format(epoch)))

    metrics = {'3d_dice': top_dice_mean, '3d_precision': precision_at_top_dice,
               '3d_recall': recall_at_top_dice, '3d_volume': volume_at_top_dice,
               '3d_surface_dice': surface_dice_at_top_dice, '3d_hd95': hd95_at_top_dice,
               '3d_hd100':hd100_at_top_dice, '3d_abs_volume': abs_volume_at_top_dice}

    return metrics


def getMetricsFindThreshold(config, wandb_run_dir, epoch, best_up_to_now=0):
    # Load sweeps outputs for val. set
    sweep_out_dir = os.path.join(wandb_run_dir, 'predictions/*{}'.format(config.file_extension))
    sweep_out_files = glob.glob(sweep_out_dir)
    mask_files = getValData(config, validation=True)
    mask_files_complete = getValData(config, validation=True, complete=True)
    assert (len(mask_files) == len(sweep_out_files))
    # generate dataframes

    dice_df = getScoreDataframe(mask_files, sweep_out_files, mask_files_complete, metric='dice')

    # remove area for metric calculation
    df2 = dice_df.drop(columns='area')
    # find best avg. threshold
    max_th = df2.groupby('case').mean().mean(axis=0).idxmax(axis=0)
    prec_df = getScoreDataframe(mask_files, sweep_out_files, mask_files_complete, metric='precision', threshold=max_th)
    rec_df = getScoreDataframe(mask_files, sweep_out_files, mask_files_complete, metric='recall', threshold=max_th)
    vol_df = getScoreDataframe(mask_files, sweep_out_files, mask_files_complete, metric='volume', threshold=max_th)
    abs_vol_df = getScoreDataframe(mask_files, sweep_out_files, mask_files_complete, metric='abs_volume', threshold=max_th)
    surface_dice_df = getScoreDataframe(mask_files, sweep_out_files, mask_files_complete, metric='surface_dice', threshold=max_th)
    hd95_df = getScoreDataframe(mask_files, sweep_out_files, mask_files_complete, metric='hd95', threshold=max_th)
    hd100_df = getScoreDataframe(mask_files, sweep_out_files, mask_files_complete, metric='hd100', threshold=max_th)

    hd95_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    hd100_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    surface_dice_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    surface_dice_at_top_dice = surface_dice_df[max_th].mean()
    hd95_at_top_dice = hd95_df[max_th].mean()
    hd100_at_top_dice = hd100_df[max_th].mean()

    precision_at_top_dice = prec_df[max_th].mean()
    recall_at_top_dice = rec_df[max_th].mean()
    volume_at_top_dice = vol_df[max_th].mean()
    abs_volume_at_top_dice = abs_vol_df[max_th].mean()

    df2 = dice_df.drop(columns='area')
    top_dice_per_case = df2.groupby('case')[max_th].mean()
    top_dice_mean = top_dice_per_case.mean()

    if top_dice_mean >= best_up_to_now:
        # save dataframes
        os.makedirs(os.path.join(wandb_run_dir, 'pickles'), exist_ok=True)
        dice_df.to_pickle(os.path.join(wandb_run_dir, 'pickles', 'dice-{}'.format(epoch)))
        prec_df.to_pickle(os.path.join(wandb_run_dir, 'pickles', 'precision-{}'.format(epoch)))
        rec_df.to_pickle(os.path.join(wandb_run_dir, 'pickles', 'recall-{}'.format(epoch)))
        vol_df.to_pickle(os.path.join(wandb_run_dir, 'pickles', 'volume-{}'.format(epoch)))
        surface_dice_df.to_pickle(os.path.join(wandb_run_dir, 'pickles', 'surface_dice-{}'.format(epoch)))
        hd95_df.to_pickle(os.path.join(wandb_run_dir, 'pickles', 'hd95-{}'.format(epoch)))
        hd100_df.to_pickle(os.path.join(wandb_run_dir, 'pickles', 'hd100-{}'.format(epoch)))
        abs_vol_df.to_pickle(os.path.join(wandb_run_dir, 'pickles', 'absvol-{}'.format(epoch)))

    metrics = {'3d_dice': top_dice_mean, '3d_precision': precision_at_top_dice,
               '3d_recall': recall_at_top_dice, '3d_volume': volume_at_top_dice,
               '3d_surface_dice': surface_dice_at_top_dice, '3d_hd95': hd95_at_top_dice,
               '3d_hd100': hd100_at_top_dice, '3d_abs_volume': abs_volume_at_top_dice, 'final_threshold': max_th}
    return metrics
def getValData(config, validation=None, complete=False):
    # Load masks for val set
    mask_dir = os.path.join(config.data_folder, 'MASK')
    if complete:
        mask_dir = os.path.join(config.data_folder, 'COMPLETE_MASK')
    fold_file = os.path.join(config.data_folder, config.fold + '.txt')
    validation_cases = np.loadtxt(fold_file, delimiter=",")
    validation_cases = [str(int(x)).zfill(2) for x in validation_cases]
    if complete:
        mask_name = glob.glob(mask_dir + '{}*{}'.format(os.sep, '.nii'))
    else:
        mask_name = glob.glob(mask_dir + '{}*{}'.format(os.sep, config.file_extension))
    if validation:
        if not complete:
            mask_name = [x for x in mask_name if x.split(os.sep)[-1].split("_")[1] in validation_cases]
        else:
            mask_name = [x for x in mask_name if x.split(os.sep)[-1].split("_")[1].split('.')[0] in validation_cases]
    else:
        if not complete:
            mask_name = [x for x in mask_name if x.split(os.sep)[-1].split("_")[1] not in validation_cases]
        else:
            mask_name = [x for x in mask_name if x.split(os.sep)[-1].split("_")[1].split('.')[0] not in validation_cases]
    return mask_name

