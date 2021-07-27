import os
import datetime
import numpy as np
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
from utils import *

OUT_DIR = './viz/'

norm = matplotlib.colors.Normalize(vmin=0.0, vmax=255.0)
cm = 1/2.54
dpi = 300

def denormalization(x, norm_mean, norm_std):
    mean = np.array(norm_mean)
    std = np.array(norm_std)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def export_hist(c, gts, scores, threshold):
    print('Exporting histogram...')
    plt.rcParams.update({'font.size': 4})
    image_dirs = os.path.join(OUT_DIR, c.model)
    os.makedirs(image_dirs, exist_ok=True)
    Y = scores.flatten()
    Y_label = gts.flatten()
    fig = plt.figure(figsize=(4*cm, 4*cm), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    plt.hist([Y[Y_label==1], Y[Y_label==0]], 500, density=True, color=['r', 'g'], label=['ANO', 'TYP'], alpha=0.75, histtype='barstacked')
    image_file = os.path.join(image_dirs, 'hist_images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    fig.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()

def export_groundtruth(c, test_img, gts):
    image_dirs = os.path.join(OUT_DIR, c.model, 'gt_images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    # images
    if not os.path.isdir(image_dirs):
        print('Exporting grountruth...')
        os.makedirs(image_dirs, exist_ok=True)
        num = len(test_img)
        kernel = morphology.disk(4)
        for i in range(num):
            img = test_img[i]
            img = denormalization(img, c.norm_mean, c.norm_std)
            # gts
            gt_mask = gts[i].astype(np.float64)
            gt_mask = morphology.opening(gt_mask, kernel)
            gt_mask = (255.0*gt_mask).astype(np.uint8)
            gt_img = mark_boundaries(img, gt_mask, color=(1, 0, 0), mode='thick')
            #
            fig = plt.figure(figsize=(2*cm, 2*cm), dpi=dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(gt_img)
            image_file = os.path.join(image_dirs, '{:08d}'.format(i))
            fig.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
            plt.close()


def export_scores(c, test_img, scores, threshold):
    image_dirs = os.path.join(OUT_DIR, c.model, 'sc_images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    # images
    if not os.path.isdir(image_dirs):
        print('Exporting scores...')
        os.makedirs(image_dirs, exist_ok=True)
        num = len(test_img)
        kernel = morphology.disk(4)
        scores_norm = 1.0/scores.max()
        for i in range(num):
            img = test_img[i]
            img = denormalization(img, c.norm_mean, c.norm_std)
            # scores
            score_mask = np.zeros_like(scores[i])
            score_mask[scores[i] >  threshold] = 1.0
            score_mask = morphology.opening(score_mask, kernel)
            score_mask = (255.0*score_mask).astype(np.uint8)
            score_img = mark_boundaries(img, score_mask, color=(1, 0, 0), mode='thick')
            score_map = (255.0*scores[i]*scores_norm).astype(np.uint8)
            #
            fig_img, ax_img = plt.subplots(2, 1, figsize=(2*cm, 4*cm))
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
                ax_i.spines['top'].set_visible(False)
                ax_i.spines['right'].set_visible(False)
                ax_i.spines['bottom'].set_visible(False)
                ax_i.spines['left'].set_visible(False)
            #
            plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
            ax_img[0].imshow(img, cmap='gray', interpolation='none')
            ax_img[0].imshow(score_map, cmap='jet', norm=norm, alpha=0.5, interpolation='none')
            ax_img[1].imshow(score_img)
            image_file = os.path.join(image_dirs, '{:08d}'.format(i))
            fig_img.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
            plt.close()


def export_test_images(c, test_img, gts, scores, threshold):
    image_dirs = os.path.join(OUT_DIR, c.model, 'images_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    cm = 1/2.54
    # images
    if not os.path.isdir(image_dirs):
        print('Exporting images...')
        os.makedirs(image_dirs, exist_ok=True)
        num = len(test_img)
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 8}
        kernel = morphology.disk(4)
        scores_norm = 1.0/scores.max()
        for i in range(num):
            img = test_img[i]
            img = denormalization(img, c.norm_mean, c.norm_std)
            # gts
            gt_mask = gts[i].astype(np.float64)
            print('GT:', i, gt_mask.sum())
            gt_mask = morphology.opening(gt_mask, kernel)
            gt_mask = (255.0*gt_mask).astype(np.uint8)
            gt_img = mark_boundaries(img, gt_mask, color=(1, 0, 0), mode='thick')
            # scores
            score_mask = np.zeros_like(scores[i])
            score_mask[scores[i] >  threshold] = 1.0
            print('SC:', i, score_mask.sum())
            score_mask = morphology.opening(score_mask, kernel)
            score_mask = (255.0*score_mask).astype(np.uint8)
            score_img = mark_boundaries(img, score_mask, color=(1, 0, 0), mode='thick')
            score_map = (255.0*scores[i]*scores_norm).astype(np.uint8)
            #
            fig_img, ax_img = plt.subplots(3, 1, figsize=(2*cm, 6*cm))
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
                ax_i.spines['top'].set_visible(False)
                ax_i.spines['right'].set_visible(False)
                ax_i.spines['bottom'].set_visible(False)
                ax_i.spines['left'].set_visible(False)
            #
            plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
            ax_img[0].imshow(gt_img)
            ax_img[1].imshow(score_map, cmap='jet', norm=norm)
            ax_img[2].imshow(score_img)
            image_file = os.path.join(image_dirs, '{:08d}'.format(i))
            fig_img.savefig(image_file, dpi=dpi, format='svg', bbox_inches = 'tight', pad_inches = 0.0)
            plt.close()
