#!/usr/bin/env python3
"""Fine-tune Wnet against a density-gated blend of two frozen teachers."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

FEATURES = ['T', 'AIRD', 'U', 'V', 'W', 'KM', 'RI', 'QV', 'QI', 'QL',
            'AIRD_sfc', 'KM_sfc', 'RI_sfc', 'QV_sfc']
MEAN = np.array([243.9, .6, 6.3, .013, .0002, 5.04, 21.8, .002,
                 9.75e-7, 7.87e-6, .6, 5.04, 21.8, .002], dtype='float32')
STD = np.array([30.3, .42, 16.1, 7.9, .05, 20.6, 20.8, .0036,
                7.09e-6, 2.7e-5, .42, 20.6, 20.8, .0036], dtype='float32')


def density_gate(rho, start, full):
    """Zero at rho >= start, one at rho <= full; rho in kg m-3."""
    if not np.isfinite([start, full]).all() or not start > full > 0:
        raise ValueError('Require rho-start > rho-full > 0')
    u = np.clip((start - rho) / (start - full), 0, 1)
    return u * u * (3 - 2 * u)


def expand_files(patterns):
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError('No files match: ' + pattern)
        files.extend(str(Path(f).resolve()) for f in matches)
    return sorted(set(files))


def input_blocks(path, standardized=False, surface_level=None):
    """NPZ: X[N,14]. NetCDF: named, collocated features, one time at a time."""
    if path.endswith('.npz'):
        with np.load(path, allow_pickle=False) as data:
            yield np.asarray(data['X'], dtype='float32')
        return
    if standardized:
        raise ValueError('--standardized is supported only for NPZ inputs')
    import xarray as xr
    with xr.open_dataset(path) as ds:
        for t in range(ds.sizes.get('time', 1)):
            block = ds.isel(time=t, drop=True) if 'time' in ds.dims else ds
            arrays = []
            for name in FEATURES:
                if name in block:
                    value = block[name]
                elif name.endswith('_sfc') and surface_level is not None:
                    value = block[name[:-4]].sel(lev=surface_level, drop=True)
                else:
                    raise ValueError(f'{path}: missing {name}; supply surface fields '
                                     'or --surface-level with a lev coordinate value')
                arrays.append(value)
            arrays = xr.broadcast(*arrays)
            dims = arrays[0].dims
            yield np.column_stack([v.transpose(*dims).values.reshape(-1)
                                   for v in arrays]).astype('float32')


def batches(files, args, shuffle=False, seed=0):
    rng = np.random.default_rng(seed)
    order = list(files)
    if shuffle:
        rng.shuffle(order)
    for path in order:
        for raw in input_blocks(path, args.standardized, args.surface_level):
            if raw.ndim != 2 or raw.shape[1] != 14:
                raise ValueError(f'{path}: expected X with shape (N,14), got {raw.shape}')
            rho = raw[:, 1] * STD[1] + MEAN[1] if args.standardized else raw[:, 1]
            good = np.isfinite(raw).all(axis=1) & (rho > 0)
            if not good.all():
                raise ValueError(f'{path}: invalid features or nonpositive density')
            x = raw if args.standardized else (raw - MEAN) / STD
            alpha = density_gate(rho, args.rho_start, args.rho_full)
            idx = np.arange(len(x))
            if shuffle:
                rng.shuffle(idx)
            for i in range(0, len(idx), args.batch_size):
                j = idx[i:i + args.batch_size]
                yield x[j], alpha[j, None].astype('float32')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wnet', default='Wnet.h5')
    p.add_argument('--prior', default='data/Wnet_prior.h5')
    p.add_argument('--train', nargs='+', required=True, help='NPZ/NetCDF paths or quoted globs')
    p.add_argument('--validation', nargs='+', required=True, help='Separate held-out files')
    p.add_argument('--rho-start', type=float, required=True, help='Start relaxation, kg m-3')
    p.add_argument('--rho-full', type=float, required=True, help='Full prior target, kg m-3')
    p.add_argument('--surface-level', type=float, help='NetCDF lev coordinate for surface features')
    p.add_argument('--standardized', action='store_true', help='NPZ X is already standardized')
    p.add_argument('--output-dir', default='stabilization/run')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--scale-floor', type=float, default=.1, help='Relative loss floor, m s-1')
    p.add_argument('--trainable-dense', type=int, default=2,
                   help='Train last N Dense layers; 0 trains all layers')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    density_gate(np.array([1.]), args.rho_start, args.rho_full)
    if min(args.epochs, args.patience, args.batch_size) <= 0 or args.trainable_dense < 0:
        p.error('Epochs, patience and batch size must be positive; trainable-dense >= 0')
    if not np.isfinite([args.learning_rate, args.scale_floor]).all() or min(
            args.learning_rate, args.scale_floor) <= 0:
        p.error('learning-rate and scale-floor must be finite and positive')
    train, val = expand_files(args.train), expand_files(args.validation)
    if set(train) & set(val):
        p.error('Training and validation files must not overlap')
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        p.error('Output directory is not empty; use a new run directory')

    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    teacher = tf.keras.models.load_model(args.wnet, compile=False)
    prior = tf.keras.models.load_model(args.prior, compile=False)
    student = tf.keras.models.clone_model(teacher)
    student.set_weights(teacher.get_weights())
    for model in (teacher, prior, student):
        if tuple(model.input_shape[1:]) != (14,) or tuple(model.output_shape[1:]) != (1,):
            raise ValueError('Models must accept 14 standardized features and return one sigmaW')
    teacher.trainable = prior.trainable = False
    student.trainable = True
    dense = [layer for layer in student.layers if isinstance(layer, tf.keras.layers.Dense)]
    if args.trainable_dense > len(dense):
        p.error('trainable-dense exceeds the number of Dense layers')
    selected = dense[-args.trainable_dense:] if args.trainable_dense else None
    for layer in student.layers:
        layer.trainable = selected is None or layer in selected
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.)

    def references(x, alpha):
        old, base = teacher(x, training=False), prior(x, training=False)
        tf.debugging.assert_all_finite(old, 'Nonfinite Wnet reference')
        tf.debugging.assert_all_finite(base, 'Nonfinite prior reference')
        target = (1 - alpha) * old + alpha * base
        tf.debugging.assert_greater_equal(target, tf.zeros_like(target),
                                         message='Negative sigmaW target; inspect reference models')
        return old, base, tf.stop_gradient(target)

    @tf.function
    def train_step(x, alpha):
        _, _, target = references(x, alpha)
        with tf.GradientTape() as tape:
            pred = student(x, training=True)
            loss = tf.reduce_mean(tf.square((pred - target) /
                                           tf.maximum(target, args.scale_floor)))
            tf.debugging.assert_all_finite(loss, 'Nonfinite training loss')
        grads = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(grads, student.trainable_variables))
        return loss

    def evaluate():
        totals = {name: np.zeros(5) for name in ('preserve', 'transition', 'prior')}
        loss_sum, n = 0., 0
        for x, a in batches(val, args):
            old, base, target = [v.numpy() for v in references(x, a)]
            pred = student(x, training=False).numpy()
            if not np.isfinite(pred).all():
                raise ValueError('Nonfinite validation predictions')
            loss_sum += np.square((pred - target) / np.maximum(target, args.scale_floor)).sum()
            n += len(x)
            for name, mask in [('preserve', a[:, 0] == 0),
                               ('transition', (a[:, 0] > 0) & (a[:, 0] < 1)),
                               ('prior', a[:, 0] == 1)]:
                totals[name] += [mask.sum(), np.abs(pred[mask] - target[mask]).sum(),
                                np.abs(pred[mask] - old[mask]).sum(),
                                np.abs(pred[mask] - base[mask]).sum(), (pred[mask] < 0).sum()]
        if not n:
            raise ValueError('Empty validation dataset')
        metrics = {'loss': float(loss_sum / n)}
        for name, (count, target_err, old_err, prior_err, negative) in totals.items():
            metrics[name] = {'count': int(count), 'target_mae': float(target_err / count) if count else None,
                             'wnet_mae': float(old_err / count) if count else None,
                             'prior_mae': float(prior_err / count) if count else None,
                             'negative_count': int(negative)}
        return metrics

    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps({**vars(args), 'features': FEATURES,
        'mean': MEAN.tolist(), 'std': STD.tolist(), 'train_files': train,
        'validation_files': val}, indent=2))
    baseline = evaluate()
    if any(baseline[k]['count'] == 0 for k in ('preserve', 'transition', 'prior')):
        raise ValueError('Validation must include preservation, transition and full-prior regimes')
    best, stale = baseline['loss'], 0
    student.save(str(out / 'Wnet_stabilized.h5'), include_optimizer=False)
    with (out / 'history.jsonl').open('w') as history:
        history.write(json.dumps({'epoch': 0, 'validation': baseline}) + '\n')
        for epoch in range(1, args.epochs + 1):
            total, count = 0., 0
            regimes = np.zeros(3, dtype=int)
            for x, a in batches(train, args, shuffle=True, seed=args.seed + epoch):
                total += float(train_step(x, a)) * len(x)
                count += len(x)
                regimes += [(a == 0).sum(), ((a > 0) & (a < 1)).sum(), (a == 1).sum()]
            if not count or (regimes == 0).any():
                raise ValueError('Training must include all three density regimes')
            metrics = evaluate()
            record = {'epoch': epoch, 'train_loss': total / count, 'validation': metrics}
            print(json.dumps(record), flush=True)
            history.write(json.dumps(record) + '\n')
            history.flush()
            if metrics['loss'] < best:
                best, stale = metrics['loss'], 0
                student.save(str(out / 'Wnet_stabilized.h5'), include_optimizer=False)
            else:
                stale += 1
            if stale >= args.patience:
                break
    print('Best validation checkpoint:', out / 'Wnet_stabilized.h5')


if __name__ == '__main__':
    main()
