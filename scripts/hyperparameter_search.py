"""Hyperparameter search for the LPBF PINN using Optuna.

Example::

    python scripts/hyperparameter_search.py --config data/params.yaml \
        --n-trials 20 --output-dir data/hpo

Optuna is required (``pip install optuna``).
"""

import argparse
import copy
import json
from pathlib import Path

import yaml


def load_base_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_trial_config(base_config, trial):
    """Create a trial configuration by sampling hyperparameters."""
    cfg = copy.deepcopy(base_config)

    # Network architecture
    cfg['model']['hidden_width'] = trial.suggest_categorical(
        'hidden_width', [64, 128, 256, 512]
    )
    cfg['model']['hidden_depth'] = trial.suggest_int('hidden_depth', 2, 6)
    cfg['model']['dropout_rate'] = trial.suggest_float(
        'dropout_rate', 0.0, 0.3, step=0.05
    )

    # Optimizer
    cfg['training']['optimizer']['learning_rate'] = trial.suggest_float(
        'learning_rate', 1e-4, 1e-2, log=True
    )

    # Physics-loss weights
    cfg['training']['lambda_heat'] = trial.suggest_float('lambda_heat', 0.0, 1.0)
    cfg['training']['lambda_stress'] = trial.suggest_float('lambda_stress', 0.0, 1.0)
    cfg['training']['lambda_porosity'] = trial.suggest_float('lambda_porosity', 0.0, 0.5)
    cfg['training']['lambda_geometry'] = trial.suggest_float('lambda_geometry', 0.0, 0.5)

    # Reduce epochs for HPO speed
    cfg['training']['n_epochs'] = cfg['training'].get('hpo_epochs', 10)
    cfg['training']['print_freq'] = cfg['training']['n_epochs'] + 1  # silence

    return cfg


def objective_factory(base_config, num_threads=1):
    """Return an Optuna objective function."""
    from pinn.train import PINNTrainer

    def objective(trial):
        cfg = make_trial_config(base_config, trial)
        try:
            trainer = PINNTrainer(config_path=None, config=cfg, num_threads=num_threads)
            trainer.load_data()
            for epoch in range(cfg['training']['n_epochs']):
                trainer.train_epoch(epoch)
                val_loss = trainer.validate()
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return val_loss
        except RuntimeError as e:
            # Bad hyperparameter combinations may fail to converge
            raise optuna.TrialPruned() from e

    return objective


def main():
    global optuna
    try:
        import optuna
        from optuna.visualization import plot_optimization_history, plot_param_importances
    except ImportError as e:
        raise ImportError(
            "Optuna is required for hyperparameter search. "
            "Install it with: pip install optuna"
        ) from e

    parser = argparse.ArgumentParser(
        description='Hyperparameter search for LPBF PINN'
    )
    parser.add_argument('--config', type=str, default='data/params.yaml',
                        help='Path to base configuration file')
    parser.add_argument('--n-trials', type=int, default=20,
                        help='Number of Optuna trials')
    parser.add_argument('--output-dir', type=str, default='data/hpo',
                        help='Directory to save study results')
    parser.add_argument('--num-threads', type=int, default=1,
                        help='Number of PyTorch threads per trial')
    args = parser.parse_args()

    base_config = load_base_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        objective_factory(base_config, num_threads=args.num_threads),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Save best config
    best_config = make_trial_config(base_config, study.best_trial)
    with open(output_dir / 'best_config.yaml', 'w') as f:
        yaml.dump(best_config, f)

    # Save study summary
    summary = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
    }
    with open(output_dir / 'study_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save plots
    try:
        fig = plot_optimization_history(study)
        fig.write_image(str(output_dir / 'optimization_history.png'))
        fig = plot_param_importances(study)
        fig.write_image(str(output_dir / 'param_importances.png'))
    except Exception as e:
        print(f"Could not generate HPO plots: {e}")

    print(f"Best validation loss: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
