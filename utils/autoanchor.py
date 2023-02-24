# Auto-anchor utils

import numpy as np
import torch
import yaml
from scipy.cluster.vq import kmeans
from tqdm import tqdm

from utils.general import colorstr

# 检查anchor顺序和stride顺序是否一致
# check anchors to make sure they are aligned to your stride order.
# i.e. they should both be large to small or small to large depending on your head.
def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


# 自动计算最佳锚框(AutoAnchor)
"""
计算默认锚框anchor与数据集标签框的宽高比值
标签的高h宽w与anchor的高h_a宽w_a的比值, 即h/h_a, w/w_a都要在(1/hyp['anchor_t'], hyp['anchor_t'])是可以接受的
如果bpr(best possible recall)小于98%，则根据k-means算法聚类新的锚框。
Autoanchor only runs when the best possible recall (BPR) is under threshold(0.98)
"""

def check_anchors(dataset, model, thr=4.0, imgsz=640): #thr阈值 来自于超参hyp['anchor_t']
    # Check anchor fit to data, recompute if necessary
    prefix = colorstr('autoanchor: ')
    print(f'\n{prefix}Analyzing anchors... ', end='')
    #取出model的最后一层
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect() 
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    # 考虑到数据有随机增强，对dataset的所有bbox进行随机变换，
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        r = wh[:, None] / k[None] # nx9x2 每个grid有9个achor
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric # ratio metric; # nx9 wh方向找最小值（即w或h那个方向最小）
        best = x.max(1)[0]  # best_x  nx9个anchor中找最大值
        # 比例值大于阈值，则认为匹配 sum(1) 当axis=1的时候，求的是每一行元素的和
        aat = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold 判断x>0.75,求和求均值
        bpr = (best > 1. / thr).float().mean()  # best possible recall 判断best>1 求均值 (最大可能召回率)
        return bpr, aat

    anchors = m.anchor_grid.clone().cpu().view(-1, 2)  # current anchors # 列数等于2，行数自动推理
    bpr, aat = metric(anchors)
    print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}', end='')
     # threshold to recompute；如果bpr小于98%，则根据k-means算法聚类新的锚框
    if bpr < 0.98:  # threshold to recompute
        print('. Attempting to improve anchors, please wait...')
        na = m.anchor_grid.numel() // 2  # number of anchors
        try: # 重新计算anchor
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            print(f'{prefix}ERROR: {e}')
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchor_grid[:] = anchors.clone().view_as(m.anchor_grid)  # for inference
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            check_anchor_order(m)
            print(f'{prefix}New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print(f'{prefix}Original anchors better than new anchors. Proceeding with original anchors.')
    print('')  # newline

# 对anchors做kmeans聚类
def kmean_anchors(path='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: path to dataset *.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    thr = 1. / thr 
    prefix = colorstr('autoanchor: ')
    
    #计算指标函数
    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]  #传进的图片大小/achor大小
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric #求r与1/r的最小值
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    #变异深度
    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness 

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print(f'{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr')
        print(f'{prefix}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, '
              f'past_thr={x[x > thr].mean():.3f}-mean: ', end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    if isinstance(path, str):  # *.yaml file
        with open(path) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
    else:
        dataset = path  # dataset

    # Get label wh 获得数据集中图片的shape
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh 未做随机化尺度变化

    # Filter 查看目标框的WH的值是否小于3
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(f'{prefix}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels
    # wh = wh * (np.random.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans calculation
    print(f'{prefix}Running kmeans for {n} anchors on {len(wh)} points...')
    s = wh.std(0)  # sigmas for whitening 白化
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    assert len(k) == n, print(f'{prefix}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}')
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unfiltered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'{prefix}Evolving anchors with Genetic Algorithm:')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f: # 有更好的fitness值
            f, k = fg, kg.copy() # 使用进化后的锚框值
            pbar.desc = f'{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k)

    return print_results(k)
