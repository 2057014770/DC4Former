# Copyright (c) OpenMMLab. All rights reserved.
# Unified rotation-wise mIoU evaluation script.
# The evaluation path is kept consistent across configured datasets:
#   1) fixed-canvas rotation by default (auto_bound=False)
#   2) update ori_shape/img_shape/pad_shape consistently
#   3) rotate every segmentation field
#   4) reset scale_factor after the evaluation rotation
#   5) accumulate the confusion matrix image by image
#   6) raise an error on prediction/GT shape mismatch by default
#   7) print per-angle per-class IoU and foreground-mIoU summaries
#
# Default angles:
#   0,45,90,135,180,225,270,315

import argparse
import copy
import csv
import gc
import importlib
import json
import math
import os
import os.path as osp
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmseg.registry import TRANSFORMS


try:
    from mmseg.utils import register_all_modules
    register_all_modules(init_default_scope=True)
except Exception:
    pass


@TRANSFORMS.register_module(force=True)
class RotationEvalTransform(BaseTransform):
    """Rotate image and GT for rotation-wise evaluation."""

    def __init__(
        self,
        angle=0,
        ignore_index=255,
        image_border_value=0,
        auto_bound=False,
        update_ori_shape=False,
        rotate_all_seg_fields=False,
        reset_scale_factor=False,
    ):
        self.angle = float(angle) % 360.0
        self.ignore_index = int(ignore_index)
        self.image_border_value = image_border_value
        self.auto_bound = bool(auto_bound)
        self.update_ori_shape = bool(update_ori_shape)
        self.rotate_all_seg_fields = bool(rotate_all_seg_fields)
        self.reset_scale_factor = bool(reset_scale_factor)

    @staticmethod
    def _is_right_angle(angle):
        return abs(angle % 90.0) < 1e-6

    def _rotate_img(self, img):
        if self.angle == 0:
            return img

        if self._is_right_angle(self.angle) and not self.auto_bound:
            k = int(round(self.angle / 90.0)) % 4
            return np.ascontiguousarray(np.rot90(img, k=k))

        return mmcv.imrotate(
            img,
            self.angle,
            border_value=self.image_border_value,
            interpolation='bilinear',
            auto_bound=self.auto_bound)

    def _rotate_seg(self, seg):
        if self.angle == 0:
            return seg

        if self._is_right_angle(self.angle) and not self.auto_bound:
            k = int(round(self.angle / 90.0)) % 4
            return np.ascontiguousarray(np.rot90(seg, k=k))

        return mmcv.imrotate(
            seg,
            self.angle,
            border_value=self.ignore_index,
            interpolation='nearest',
            auto_bound=self.auto_bound)

    def transform(self, results):
        if self.angle == 0:
            return results

        results['img'] = self._rotate_img(results['img'])
        results['img_shape'] = results['img'].shape[:2]

        if self.rotate_all_seg_fields:
            seg_keys = list(results.get('seg_fields', []))
            if 'gt_seg_map' in results and 'gt_seg_map' not in seg_keys:
                seg_keys.append('gt_seg_map')

            for key in seg_keys:
                if key in results:
                    results[key] = self._rotate_seg(results[key])
        else:
            if 'gt_seg_map' in results:
                results['gt_seg_map'] = self._rotate_seg(results['gt_seg_map'])

        if self.update_ori_shape:
            if 'gt_seg_map' in results:
                eval_shape = results['gt_seg_map'].shape[:2]
            else:
                eval_shape = results['img'].shape[:2]

            results['ori_shape'] = eval_shape
            results['img_shape'] = results['img'].shape[:2]

            if 'pad_shape' in results:
                results['pad_shape'] = results['img'].shape[:2]

            if self.reset_scale_factor and 'scale_factor' in results:
                results['scale_factor'] = (1.0, 1.0)

        else:
            if self.auto_bound and 'ori_shape' in results:
                h, w = results['ori_shape'][:2]
                dummy = np.zeros((h, w), dtype=np.uint8)
                dummy_rot = mmcv.imrotate(dummy, self.angle, auto_bound=True)
                results['ori_shape'] = dummy_rot.shape[:2]

        results['rotation_eval_angle'] = self.angle
        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rotation-wise mIoU evaluation with per-class IoU tables')

    parser.add_argument('config')
    parser.add_argument('checkpoint')

    parser.add_argument('--work-dir', default=None)


    parser.add_argument(
        '--num-classes',
        type=int,
        default=None,
        help='Dataset-specific class count. If omitted, the script tries '
             'to infer it from the config.')

    parser.add_argument('--ignore-index', type=int, default=255)

    parser.add_argument(
        '--angles',
        type=str,
        default='0,45,90,135,180,225,270,315')

    parser.add_argument(
        '--background-index',
        type=int,
        default=0,
        help='Background class index. Final summary uses fg mIoU excluding this class.')

    parser.add_argument(
        '--auto-bound',
        action='store_true',
        help='Use expanded-canvas rotation (auto_bound=True). Default is fixed-canvas False.')

    parser.add_argument(
        '--insert-after-loadann',
        action='store_true',
        help='Insert rotation after LoadAnnotations. Default inserts before PackSegInputs.')

    parser.add_argument('--image-border-value', type=int, default=0)

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Default 1 for safe mixed-size evaluation.')

    parser.add_argument(
        '--shape-mismatch',
        choices=['resize_gt_to_pred', 'resize_pred_to_gt', 'error'],
        default='error',
        help='Default error prevents silent label/prediction resizing in formal evaluation.')

    parser.add_argument(
        '--class-names',
        type=str,
        default=None,
        help='Optional comma-separated class names. If omitted, try config metainfo.')

    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--print-pipeline', action='store_true')
    parser.add_argument('--print-shapes', action='store_true')

    parser.add_argument('--official-zero-miou', type=float, default=None)
    parser.add_argument('--zero-tol', type=float, default=0.05)

    parser.add_argument(
        '--custom-imports',
        nargs='+',
        default=None,
        help='Optional custom modules to import before building model.')

    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def import_custom_modules(module_names):
    if module_names is None:
        return
    for module_name in module_names:
        print(f'[Custom Import] {module_name}')
        importlib.import_module(module_name)


def parse_angles(s):
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def get_core_model(runner):
    model = runner.model
    return model.module if hasattr(model, 'module') else model


def get_dataset_cfgs(dataset_cfg):
    out = []

    if isinstance(dataset_cfg, dict):
        if 'pipeline' in dataset_cfg:
            out.append(dataset_cfg)
        elif 'dataset' in dataset_cfg:
            out.extend(get_dataset_cfgs(dataset_cfg['dataset']))
        elif 'datasets' in dataset_cfg:
            for sub in dataset_cfg['datasets']:
                out.extend(get_dataset_cfgs(sub))
        else:
            raise RuntimeError('Cannot find pipeline in cfg.test_dataloader.dataset.')
    else:
        raise TypeError(f'Unsupported dataset cfg type: {type(dataset_cfg)}')

    return out


def infer_num_classes(cfg):
    try:
        decode_head = cfg.model.decode_head
        if isinstance(decode_head, dict) and 'num_classes' in decode_head:
            return int(decode_head['num_classes'])
        if isinstance(decode_head, (list, tuple)):
            for head in decode_head:
                if isinstance(head, dict) and 'num_classes' in head:
                    return int(head['num_classes'])
    except Exception:
        pass

    try:
        head = cfg.model.head
        if isinstance(head, dict) and 'num_classes' in head:
            return int(head['num_classes'])
    except Exception:
        pass

    try:
        for dataset_cfg in get_dataset_cfgs(cfg.test_dataloader.dataset):
            metainfo = dataset_cfg.get('metainfo', None)
            if metainfo is not None and 'classes' in metainfo:
                return len(metainfo['classes'])
    except Exception:
        pass

    return None


def infer_class_names(cfg, num_classes, arg_class_names=None):
    if arg_class_names is not None:
        names = [x.strip() for x in arg_class_names.split(',')]
        if len(names) != num_classes:
            raise ValueError(
                f'Provided --class-names has {len(names)} names, but num_classes={num_classes}.'
            )
        return names

    try:
        for dataset_cfg in get_dataset_cfgs(cfg.test_dataloader.dataset):
            metainfo = dataset_cfg.get('metainfo', None)
            if metainfo is not None and 'classes' in metainfo:
                classes = list(metainfo['classes'])
                if len(classes) == num_classes:
                    return [str(x) for x in classes]
    except Exception:
        pass

    return [f'class_{i}' for i in range(num_classes)]


def build_rotation_cfg(args, angle):
    # Apply one consistent protocol to every configured dataset.
    return dict(
        type='RotationEvalTransform',
        angle=float(angle),
        ignore_index=args.ignore_index,
        image_border_value=args.image_border_value,
        auto_bound=args.auto_bound,
        update_ori_shape=True,
        rotate_all_seg_fields=True,
        reset_scale_factor=True)


def inject_transform_into_pipeline(pipeline, rot_cfg, insert_after_loadann=False):
    pipeline = copy.deepcopy(pipeline)

    insert_idx = None

    if insert_after_loadann:
        for i, t in enumerate(pipeline):
            if t.get('type') == 'LoadAnnotations':
                insert_idx = i + 1
                break
    else:
        for i, t in enumerate(pipeline):
            if t.get('type') == 'PackSegInputs':
                insert_idx = i
                break

    if insert_idx is None:
        insert_idx = len(pipeline)

    pipeline.insert(insert_idx, rot_cfg)
    return pipeline


def inject_rotation_transform(cfg, args, angle):
    rot_cfg = build_rotation_cfg(args, angle)

    for dataset_cfg in get_dataset_cfgs(cfg.test_dataloader.dataset):
        dataset_cfg['pipeline'] = inject_transform_into_pipeline(
            dataset_cfg['pipeline'],
            rot_cfg,
            insert_after_loadann=args.insert_after_loadann)

    return cfg


def set_test_batch_size(cfg, batch_size):
    if batch_size is None:
        return cfg

    cfg.test_dataloader['batch_size'] = int(batch_size)
    return cfg


def make_base_work_dir(cfg, args):
    if args.work_dir is not None:
        return args.work_dir

    if cfg.get('work_dir', None) is not None:
        return osp.join(cfg.work_dir, 'rotation_miou')

    return './work_dirs/rotation_miou'


def angle_dir(angle):
    s = f'{angle:g}'.replace('-', 'm').replace('.', 'p')
    return f'angle_{s}'


def build_cfg_for_angle(base_cfg, args, angle, base_work_dir):
    cfg = copy.deepcopy(base_cfg)

    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint
    cfg.work_dir = osp.join(base_work_dir, angle_dir(angle))

    cfg = set_test_batch_size(cfg, args.batch_size)
    cfg = inject_rotation_transform(cfg, args, angle)

    if args.print_pipeline:
        print('\n' + '=' * 80)
        print(f'Pipeline for angle={angle:g}, protocol=rotation_eval_unified')
        print('=' * 80)

        for dataset_idx, dataset_cfg in enumerate(get_dataset_cfgs(cfg.test_dataloader.dataset)):
            print(f'\n[Dataset {dataset_idx}]')
            for i, t in enumerate(dataset_cfg['pipeline']):
                print(i, t)

    return cfg


def stack_pred(samples):
    out = []
    for s in samples:
        p = s.pred_sem_seg.data
        if p.ndim == 3:
            p = p.squeeze(0)
        out.append(p.long())
    return torch.stack(out, 0)


def stack_gt(samples, pred_shape=None):
    out = []
    for s in samples:
        g = s.gt_sem_seg.data
        if g.ndim == 3:
            g = g.squeeze(0)
        out.append(g.long())
    gt = torch.stack(out, 0)

    if pred_shape is not None and tuple(gt.shape[-2:]) != tuple(pred_shape):
        gt = F.interpolate(
            gt.unsqueeze(1).float(),
            size=pred_shape,
            mode='nearest').squeeze(1).long()

    return gt


def squeeze_label(x):
    if x.ndim == 3:
        x = x.squeeze(0)
    return x.long()


def resize_label_nearest(label_hw, target_hw):
    return F.interpolate(
        label_hw[None, None].float(),
        size=target_hw,
        mode='nearest').squeeze(0).squeeze(0).long()


def align_pred_gt_one(pred, gt, mode):
    if tuple(pred.shape[-2:]) == tuple(gt.shape[-2:]):
        return pred, gt, False

    if mode == 'error':
        raise RuntimeError(
            f'pred/gt shape mismatch: pred={tuple(pred.shape)}, gt={tuple(gt.shape)}')

    if mode == 'resize_gt_to_pred':
        gt = resize_label_nearest(gt, pred.shape[-2:])
        return pred, gt, True

    if mode == 'resize_pred_to_gt':
        pred = resize_label_nearest(pred, gt.shape[-2:])
        return pred, gt, True

    raise ValueError(f'Unknown shape_mismatch mode: {mode}')


def confusion_matrix(pred, target, num_classes, ignore_index=255):
    assert pred.shape == target.shape, f'pred {pred.shape} target {target.shape}'

    pred = pred.reshape(-1)
    target = target.reshape(-1)

    valid = (
        (target != ignore_index)
        & (target >= 0)
        & (target < num_classes)
        & (pred >= 0)
        & (pred < num_classes)
    )

    pred = pred[valid]
    target = target[valid]

    hist = torch.bincount(
        num_classes * target + pred,
        minlength=num_classes ** 2)

    return hist.reshape(num_classes, num_classes).float()


def compute_iou_stats_from_hist(hist, background_index=0):
    eps = 1e-10

    hist_cpu = hist.detach().cpu().float()

    inter = torch.diag(hist_cpu)
    pred_area = hist_cpu.sum(0)
    label_area = hist_cpu.sum(1)
    union = pred_area + label_area - inter

    valid = union > 0
    iou = torch.full_like(union, float('nan'))

    if valid.any():
        iou[valid] = inter[valid] / (union[valid] + eps)

    aacc = (inter.sum() / (label_area.sum() + eps)).item() * 100.0

    class_acc = torch.full_like(label_area, float('nan'))
    valid_acc = label_area > 0
    if valid_acc.any():
        class_acc[valid_acc] = inter[valid_acc] / (label_area[valid_acc] + eps)

    valid_idx = torch.where(valid)[0].tolist()
    miou_all = (iou[valid].mean().item() * 100.0) if len(valid_idx) > 0 else 0.0
    macc_all = (class_acc[valid_acc].mean().item() * 100.0) if valid_acc.any() else 0.0

    fg_mask = valid.clone()
    if 0 <= background_index < len(fg_mask):
        fg_mask[background_index] = False

    fg_valid_idx = torch.where(fg_mask)[0].tolist()
    fg_miou = (iou[fg_mask].mean().item() * 100.0) if len(fg_valid_idx) > 0 else 0.0

    return OrderedDict(
        mIoU_all=float(miou_all),
        fg_mIoU=float(fg_miou),
        mAcc=float(macc_all),
        aAcc=float(aacc),
        per_class_iou=iou.tolist(),
        valid_class_mask=valid.tolist(),
        fg_valid_class_mask=fg_mask.tolist()
    )


def fmt_pct(x):
    if x is None:
        return '-'
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return '-'
    return f'{x * 100.0:.2f}'


def print_per_class_iou_table(angle, class_names, per_class_iou, background_index):
    print('\n' + '-' * 92)
    print(f'Per-class IoU Table | Angle = {angle:g}°')
    print('-' * 92)
    print('| Idx | Class Name | IoU (%) | Note |')
    print('|-----|------------|---------|------|')
    for idx, (name, iou) in enumerate(zip(class_names, per_class_iou)):
        note = 'background' if idx == background_index else ''
        print(f'| {idx:>3} | {name} | {fmt_pct(iou):>7} | {note} |')
    print('-' * 92)


@torch.no_grad()
def evaluate_rotation(core, dataloader, args, num_classes, device, angle):
    hist = torch.zeros((num_classes, num_classes), device=device)
    num_batches = 0
    num_images = 0
    mismatch_fixed = 0
    shape_printed = 0

    for bi, data in enumerate(dataloader):
        if args.max_batches is not None and bi >= args.max_batches:
            break

        samples = core.test_step(data)

        for si, sample in enumerate(samples):
            pred = squeeze_label(sample.pred_sem_seg.data).to(device)
            gt = squeeze_label(sample.gt_sem_seg.data).to(device)

            pred, gt, resized = align_pred_gt_one(
                pred,
                gt,
                mode=args.shape_mismatch)

            if resized:
                mismatch_fixed += 1

            if args.print_shapes and shape_printed < 5:
                print(
                    f'[angle={angle:g} unified] batch={bi}, sample={si}, '
                    f'pred={tuple(pred.shape)}, gt={tuple(gt.shape)}, '
                    f'pred_minmax=({int(pred.min())},{int(pred.max())}), '
                    f'gt_minmax=({int(gt.min())},{int(gt.max())})')
                shape_printed += 1

            hist += confusion_matrix(pred, gt, num_classes, args.ignore_index)
            num_images += 1

        num_batches += 1

        if (bi + 1) % 20 == 0:
            print(f'[angle={angle:g}] processed {bi + 1} batches, {num_images} images')

    return hist, num_batches, num_images, mismatch_fixed


@torch.no_grad()
def evaluate_angle(base_cfg, args, angle, base_work_dir, num_classes, class_names):
    print('\n' + '=' * 80)
    print(f'Evaluating angle {angle:g} deg | protocol=rotation_eval_unified')
    print('=' * 80)

    cfg = build_cfg_for_angle(
        base_cfg,
        args,
        angle,
        base_work_dir)

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()

    core = get_core_model(runner)
    core.eval()

    device = next(core.parameters()).device

    hist, num_batches, num_images, mismatch_fixed = evaluate_rotation(
        core,
        runner.test_dataloader,
        args,
        num_classes,
        device,
        angle)

    metrics = compute_iou_stats_from_hist(
        hist,
        background_index=args.background_index)

    print(f'[angle={angle:g}] mIoU(all) = {metrics["mIoU_all"]:.4f}')
    print(f'[angle={angle:g}] fg_mIoU(no-bg) = {metrics["fg_mIoU"]:.4f}')
    print(f'[angle={angle:g}] mAcc = {metrics["mAcc"]:.4f}')
    print(f'[angle={angle:g}] aAcc = {metrics["aAcc"]:.4f}')
    print(f'[angle={angle:g}] images={num_images}, mismatch_fixed={mismatch_fixed}')

    print_per_class_iou_table(
        angle=angle,
        class_names=class_names,
        per_class_iou=metrics['per_class_iou'],
        background_index=args.background_index)

    row = OrderedDict(
        angle=float(angle),
        mIoU_all=float(metrics['mIoU_all']),
        fg_mIoU=float(metrics['fg_mIoU']),
        mAcc=float(metrics['mAcc']),
        aAcc=float(metrics['aAcc']),
        per_class_iou=metrics['per_class_iou'],
        valid_class_mask=metrics['valid_class_mask'],
        fg_valid_class_mask=metrics['fg_valid_class_mask'],
        num_batches=int(num_batches),
        num_images=int(num_images),
        mismatch_fixed=int(mismatch_fixed),
        profile='rotation_eval_unified',
        num_classes=int(num_classes),
        ignore_index=int(args.ignore_index),
        background_index=int(args.background_index),
        auto_bound=bool(args.auto_bound),
        insert_after_loadann=bool(args.insert_after_loadann),
        shape_mismatch=args.shape_mismatch)

    del runner
    del core
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


def save_summary(rows, work_dir, class_names):
    os.makedirs(work_dir, exist_ok=True)

    json_path = osp.join(work_dir, 'rotation_miou_summary.json')
    angle_csv_path = osp.join(work_dir, 'rotation_miou_summary.csv')
    class_csv_path = osp.join(work_dir, 'rotation_per_class_iou.csv')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(
            {'class_names': class_names, 'results': rows},
            f,
            indent=2,
            ensure_ascii=False)

    with open(angle_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'angle',
            'fg_mIoU',
            'mIoU_all',
            'mAcc',
            'aAcc',
            'num_batches',
            'num_images',
            'mismatch_fixed',
            'profile',
            'num_classes',
            'ignore_index',
            'background_index',
            'auto_bound',
            'insert_after_loadann',
            'shape_mismatch',
        ])

        for r in rows:
            writer.writerow([
                r['angle'],
                r['fg_mIoU'],
                r['mIoU_all'],
                r['mAcc'],
                r['aAcc'],
                r['num_batches'],
                r['num_images'],
                r['mismatch_fixed'],
                r['profile'],
                r['num_classes'],
                r['ignore_index'],
                r['background_index'],
                r['auto_bound'],
                r['insert_after_loadann'],
                r['shape_mismatch'],
            ])

    with open(class_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['angle', 'class_index', 'class_name', 'iou'])

        for r in rows:
            for idx, (name, iou) in enumerate(zip(class_names, r['per_class_iou'])):
                iou_val = '' if (iou is None or (isinstance(iou, float) and math.isnan(iou))) else iou
                writer.writerow([r['angle'], idx, name, iou_val])

    return json_path, angle_csv_path, class_csv_path


def print_report(rows, background_index=0):
    print('\n' + '=' * 110)
    print('Final Rotation-wise Summary (foreground mIoU is the main metric)')
    print('=' * 110)
    print('| Angle | fg_mIoU(no-bg) (%) | mIoU(all) (%) | mAcc (%) | aAcc (%) | Images | Profile |')
    print('|-------|---------------------|---------------|----------|----------|--------|---------|')

    angle_to_fg_miou = {}
    angle_to_all_miou = {}

    for r in rows:
        a = float(r['angle'])
        angle_to_fg_miou[a] = float(r['fg_mIoU'])
        angle_to_all_miou[a] = float(r['mIoU_all'])
        print(
            f'| {int(a):>5}° '
            f'| {r["fg_mIoU"]:>19.2f} '
            f'| {r["mIoU_all"]:>13.2f} '
            f'| {r["mAcc"]:>8.2f} '
            f'| {r["aAcc"]:>8.2f} '
            f'| {r["num_images"]:>6} '
            f'| {r["profile"]} |'
        )

    orth_angles = [0.0, 90.0, 180.0, 270.0]
    diag_angles = [45.0, 135.0, 225.0, 315.0]

    orth_fg = [angle_to_fg_miou[a] for a in orth_angles if a in angle_to_fg_miou]
    diag_fg = [angle_to_fg_miou[a] for a in diag_angles if a in angle_to_fg_miou]

    print('-' * 110)
    if orth_fg:
        print(f'Orthogonal Mean fg_mIoU: {np.mean(orth_fg):.2f}%')
        print(f'Orthogonal fg_mIoU Std:  {np.std(orth_fg, ddof=0):.4f}')
    if diag_fg:
        print(f'Diagonal Mean fg_mIoU:   {np.mean(diag_fg):.2f}%')
        print(f'Diagonal fg_mIoU Std:    {np.std(diag_fg, ddof=0):.4f}')
    if orth_fg and diag_fg:
        print(f'Orthogonal-Diagonal fg_mIoU Gap: {np.mean(orth_fg) - np.mean(diag_fg):.2f}%')

    print('=' * 110)
    print(f'Foreground mIoU excludes class index {background_index}.')
    print('=' * 110)


def main():
    args = parse_args()

    import_custom_modules(args.custom_imports)

    base_cfg = Config.fromfile(args.config)
    base_cfg.launcher = args.launcher

    if args.cfg_options is not None:
        base_cfg.merge_from_dict(args.cfg_options)

    num_classes = args.num_classes
    if num_classes is None:
        num_classes = infer_num_classes(base_cfg)

    if num_classes is None:
        raise RuntimeError(
            'Cannot infer num_classes. Please pass --num-classes manually.')

    class_names = infer_class_names(base_cfg, num_classes, args.class_names)


    if args.background_index < 0 or args.background_index >= num_classes:
        raise ValueError(
            f'background_index={args.background_index} is invalid for num_classes={num_classes}'
        )

    if args.work_dir is None:
        args.work_dir = make_base_work_dir(base_cfg, args)

    os.makedirs(args.work_dir, exist_ok=True)

    angles = parse_angles(args.angles)

    print('=' * 110)
    print('Rotation-wise mIoU Evaluation')
    print('=' * 110)
    print(f'Config:               {args.config}')
    print(f'Checkpoint:           {args.checkpoint}')
    print(f'Angles:               {angles}')
    print(f'num_classes:          {num_classes}')
    print(f'class_names:          {class_names}')
    print('protocol:             rotation_eval_unified')
    print(f'ignore_index:         {args.ignore_index}')
    print(f'background_index:     {args.background_index}')
    print(f'work_dir:             {args.work_dir}')
    print(f'auto_bound:           {args.auto_bound}')
    print(f'insert_after_loadann: {args.insert_after_loadann}')
    print(f'batch_size override:  {args.batch_size}')
    print(f'shape_mismatch:       {args.shape_mismatch}')
    print('=' * 110)

    rows = []

    for angle in angles:
        row = evaluate_angle(
            base_cfg,
            args,
            angle,
            args.work_dir,
            num_classes,
            class_names)

        rows.append(row)

        if angle % 360.0 == 0 and args.official_zero_miou is not None:
            diff = abs(row['mIoU_all'] - args.official_zero_miou)
            if diff <= args.zero_tol:
                print(f'[OK] 0-degree mIoU(all) matches official result. diff={diff:.4f}')
            else:
                print('[WARNING] 0-degree mIoU(all) differs from official tools/test.py.')
                print(f'          script 0-degree mIoU(all): {row["mIoU_all"]:.4f}')
                print(f'          official mIoU:             {args.official_zero_miou:.4f}')
                print(f'          diff:                      {diff:.4f}')

    print_report(rows, background_index=args.background_index)

    json_path, angle_csv_path, class_csv_path = save_summary(rows, args.work_dir, class_names)

    print(f'\nSummary JSON saved to:      {json_path}')
    print(f'Angle summary CSV saved to: {angle_csv_path}')
    print(f'Class IoU CSV saved to:     {class_csv_path}')


if __name__ == '__main__':
    main()
