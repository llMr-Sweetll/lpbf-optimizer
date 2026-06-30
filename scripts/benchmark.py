"""Benchmark/regression script for the default LPBF workflow.

Runs generate -> train -> optimise with a small default configuration and
produces a YAML/Markdown report with runtimes and final losses.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import yaml


def load_base_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_benchmark(config_path, output_dir, quick=True):
    """Run the end-to-end benchmark and return a results dict."""
    from generate_synthetic_data import SyntheticDataGenerator
    from optimiser.nsga3 import NSGAOptimizer
    from pinn.train import PINNTrainer

    base_config = load_base_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = copy.deepcopy(base_config)
    if quick:
        config['data']['processed_data_path'] = str(output_dir / 'benchmark_dataset.h5')
        config['training']['output_dir'] = str(output_dir / 'models')
        config['optimizer']['output_dir'] = str(output_dir / 'optimized')
        config['training']['n_epochs'] = 5
        config['optimizer']['n_gen'] = 5
        n_scan_vectors = 20
        n_points_per_vector = 64
    else:
        n_scan_vectors = 100
        n_points_per_vector = 1000

    benchmark_config_path = output_dir / 'benchmark_config.yaml'
    with open(benchmark_config_path, 'w') as f:
        yaml.dump(config, f)

    results = {
        'config_path': str(benchmark_config_path.resolve()),
        'quick': quick,
        'hardware': _hardware_info(),
    }

    # Data generation
    t0 = time.perf_counter()
    generator = SyntheticDataGenerator(str(benchmark_config_path))
    dataset_path = generator.generate(
        n_scan_vectors=n_scan_vectors,
        n_points_per_vector=n_points_per_vector,
    )
    results['data_generation'] = {
        'runtime_seconds': round(time.perf_counter() - t0, 2),
        'dataset_path': str(dataset_path),
        'n_scan_vectors': n_scan_vectors,
        'n_points_per_vector': n_points_per_vector,
    }

    # Training
    t0 = time.perf_counter()
    trainer = PINNTrainer(config_path=config_path, config=config, num_threads=1)
    trainer.load_data()
    best_val_loss = float('inf')
    for epoch in range(config['training']['n_epochs']):
        trainer.train_epoch(epoch)
        val_loss = trainer.validate()
        best_val_loss = min(best_val_loss, val_loss)
    trainer.save_checkpoint(config['training']['n_epochs'] - 1, best_val_loss)
    results['training'] = {
        'runtime_seconds': round(time.perf_counter() - t0, 2),
        'n_epochs': config['training']['n_epochs'],
        'final_val_loss': round(float(val_loss), 6),
        'best_val_loss': round(float(best_val_loss), 6),
        'final_data_loss': round(float(trainer.metrics['data_loss'][-1]), 6) if trainer.metrics['data_loss'] else None,
        'final_physics_loss': round(float(trainer.metrics['physics_loss'][-1]), 6) if trainer.metrics['physics_loss'] else None,
        'model_checkpoint': str(trainer.checkpoint_dir / 'model_epoch_5.pt' if quick else trainer.checkpoint_dir / f"model_epoch_{config['training']['n_epochs']}.pt"),
    }

    # Optimisation
    t0 = time.perf_counter()
    checkpoint_path = trainer.checkpoint_dir / 'best_model.pt'
    if not checkpoint_path.exists():
        checkpoint_path = trainer.checkpoint_dir / f"model_epoch_{config['training']['n_epochs']}.pt"
    optimiser = NSGAOptimizer(str(benchmark_config_path), str(checkpoint_path))
    res = optimiser.optimize()
    results['optimisation'] = {
        'runtime_seconds': round(time.perf_counter() - t0, 2),
        'n_gen': config['optimizer']['n_gen'],
        'n_solutions': int(len(res.X)),
        'pareto_file': str(optimiser.output_dir / 'pareto_solutions.csv'),
    }

    results['total_runtime_seconds'] = round(
        results['data_generation']['runtime_seconds']
        + results['training']['runtime_seconds']
        + results['optimisation']['runtime_seconds'],
        2,
    )
    return results


def _hardware_info():
    import platform

    import torch

    return {
        'platform': platform.platform(),
        'python': platform.python_version(),
        'torch_version': str(torch.__version__),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }


def write_report(results, output_dir):
    """Write YAML and Markdown reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'benchmark_results.yaml', 'w') as f:
        yaml.dump(results, f)

    md = [
        "# LPBF-Optimizer Benchmark Results",
        "",
        f"- **Mode:** {'quick' if results['quick'] else 'full'}",
        f"- **Total runtime:** {results['total_runtime_seconds']} s",
        f"- **Device:** {results['hardware']['device']}",
        f"- **PyTorch:** {results['hardware']['torch_version']}",
        "",
        "## Data generation",
        "",
        f"- Runtime: {results['data_generation']['runtime_seconds']} s",
        f"- Scan vectors: {results['data_generation']['n_scan_vectors']}",
        f"- Points per vector: {results['data_generation']['n_points_per_vector']}",
        "",
        "## Training",
        "",
        f"- Runtime: {results['training']['runtime_seconds']} s",
        f"- Epochs: {results['training']['n_epochs']}",
        f"- Final validation loss: {results['training']['final_val_loss']}",
        f"- Best validation loss: {results['training']['best_val_loss']}",
        f"- Final data loss: {results['training']['final_data_loss']}",
        f"- Final physics loss: {results['training']['final_physics_loss']}",
        "",
        "## Optimisation",
        "",
        f"- Runtime: {results['optimisation']['runtime_seconds']} s",
        f"- Generations: {results['optimisation']['n_gen']}",
        f"- Pareto solutions: {results['optimisation']['n_solutions']}",
        "",
        "## Raw data",
        "",
        "```json",
        json.dumps(results, indent=2),
        "```",
        "",
    ]
    with open(output_dir / 'benchmark_results.md', 'w') as f:
        f.write('\n'.join(md))


def main():
    parser = argparse.ArgumentParser(description='Benchmark the default LPBF workflow')
    parser.add_argument('--config', type=str, default='data/params.yaml',
                        help='Path to base configuration file')
    parser.add_argument('--output-dir', type=str, default='data/benchmark',
                        help='Directory to write benchmark reports')
    parser.add_argument('--full', action='store_true',
                        help='Run full-scale benchmark instead of quick mode')
    args = parser.parse_args()

    results = run_benchmark(args.config, args.output_dir, quick=not args.full)
    write_report(results, args.output_dir)

    print(f"Benchmark complete in {results['total_runtime_seconds']} s")
    print(f"Reports written to {Path(args.output_dir).resolve()}")


if __name__ == '__main__':
    main()
