"""Ablation study comparing baseline MLP, bounded PINN, and physics-informed PINN.

The study trains three variants on the same dataset and reports test-set
performance, physics residuals, and physical-bound violations.
"""
import argparse
import copy
import csv
import sys
from pathlib import Path

import yaml

# The standalone PINN modules live in src/pinn and are typically run as scripts.
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src" / "pinn"))
sys.path.insert(0, str(repo_root / "src"))

from generate_synthetic_data import SyntheticDataGenerator  # noqa: E402
from train import PINNTrainer  # noqa: E402


# Names and human-readable descriptions for the three ablation variants.
VARIANTS = {
    "baseline_mlp": {
        "apply_output_bounds": False,
        "lambdas": {
            "lambda_heat": 0.0,
            "lambda_stress": 0.0,
            "lambda_porosity": 0.0,
            "lambda_geometry": 0.0,
        },
        "description": "Same architecture, no output bounds, no physics loss",
    },
    "pinn_no_physics": {
        "apply_output_bounds": True,
        "lambdas": {
            "lambda_heat": 0.0,
            "lambda_stress": 0.0,
            "lambda_porosity": 0.0,
            "lambda_geometry": 0.0,
        },
        "description": "Same architecture, output bounds enabled, no physics loss",
    },
    "pinn_physics": {
        "apply_output_bounds": True,
        "lambdas": None,  # use base-config lambdas
        "description": "Same architecture, output bounds enabled, full physics loss",
    },
}


class AblationStudy:
    """Orchestrate a three-way ablation of the LPBF PINN surrogate."""

    def __init__(
        self,
        config_path,
        output_dir="data/optimized",
        num_threads=4,
        seed=None,
        epochs=None,
        variant=None,
        regenerate=False,
        scan_vectors=50,
        points_per_vector=64,
    ):
        """
        Initialize the ablation study.

        Args:
            config_path (str): Path to the base YAML configuration.
            output_dir (str): Directory for the final ablation report.
            num_threads (int): Number of threads for each trainer.
            seed (int, optional): Random seed. Defaults to config value.
            epochs (int, optional): If given, override ``training.n_epochs``.
            variant (str, optional): If given, run only this variant.
            regenerate (bool): If True, regenerate the synthetic dataset.
            scan_vectors (int): Number of scan vectors for synthetic data generation.
            points_per_vector (int): Spatial points per scan vector.
        """
        self.config_path = Path(config_path)
        with open(self.config_path, "r") as f:
            self.base_config = yaml.safe_load(f)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads
        self.seed = seed if seed is not None else self.base_config["training"].get("random_seed", 42)
        self.epochs = epochs
        self.variant = variant
        if self.variant is not None and self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {self.variant!r}; choose from {list(VARIANTS)}")
        self.regenerate = regenerate
        self.scan_vectors = scan_vectors
        self.points_per_vector = points_per_vector

        self.results = []

    def prepare_data(self):
        """Generate or reuse the processed HDF5 dataset."""
        processed_path = Path(self.base_config["data"]["processed_data_path"])
        if self.regenerate or not processed_path.exists():
            print(f"Generating synthetic dataset: {processed_path}")
            generator = SyntheticDataGenerator(self.config_path)
            generator.generate(
                n_scan_vectors=self.scan_vectors,
                n_points_per_vector=self.points_per_vector,
            )
        else:
            print(f"Using existing dataset: {processed_path}")

    def build_variant_configs(self):
        """Build a separate config dict for each ablation variant."""
        configs = {}
        for name, spec in VARIANTS.items():
            cfg = copy.deepcopy(self.base_config)
            cfg["model"]["apply_output_bounds"] = spec["apply_output_bounds"]

            if spec["lambdas"] is not None:
                cfg["training"].update(spec["lambdas"])

            if self.epochs is not None:
                cfg["training"]["n_epochs"] = self.epochs

            # Store variant runs under a dedicated ablation sub-directory so that
            # the top-level ``latest`` symlink remains untouched.
            base_output = Path(cfg["training"]["output_dir"])
            cfg["training"]["output_dir"] = str(base_output / "ablation" / name)

            configs[name] = cfg
        return configs

    def run_variant(self, name, config):
        """Train and evaluate a single variant."""
        print(f"\n{'='*60}")
        print(f"Running variant: {name}")
        print(f"  apply_output_bounds = {config['model']['apply_output_bounds']}")
        print(
            f"  lambdas = H={config['training'].get('lambda_heat', 0):.3f}, "
            f"S={config['training'].get('lambda_stress', 0):.3f}, "
            f"P={config['training'].get('lambda_porosity', 0):.3f}, "
            f"G={config['training'].get('lambda_geometry', 0):.3f}"
        )
        print(f"{'='*60}\n")

        trainer = PINNTrainer(
            str(self.config_path),
            config=config,
            num_threads=self.num_threads,
            seed=self.seed,
        )
        trainer.train()
        metrics = trainer.evaluate()

        result = {
            "variant": name,
            "run_dir": str(trainer.run_dir),
            **metrics,
        }

        # Always persist a per-variant CSV so partial runs can be aggregated.
        self._save_single_result(result)
        self._print_single_result(result)
        return result

    def _save_single_result(self, result):
        """Write one variant's result to a dedicated CSV file."""
        csv_path = self.output_dir / f"ablation_results_{result['variant']}.csv"
        fieldnames = [
            "variant",
            "test_mse_total",
            "test_mse_stress",
            "test_mse_porosity",
            "test_mse_accuracy",
            "test_rmse_stress",
            "test_rmse_porosity",
            "test_rmse_accuracy",
            "test_r2_stress",
            "test_r2_porosity",
            "test_r2_accuracy",
            "physics_residual_total",
            "physics_residual_heat",
            "physics_residual_stress",
            "physics_residual_porosity",
            "physics_residual_geometry",
            "bound_violations_pct",
        ]
        row = {
            "variant": result["variant"],
            "test_mse_total": result["test_mse_total"],
            "test_mse_stress": result["test_mse_per_output"][0],
            "test_mse_porosity": result["test_mse_per_output"][1],
            "test_mse_accuracy": result["test_mse_per_output"][2],
            "test_rmse_stress": result["test_rmse_per_output"][0],
            "test_rmse_porosity": result["test_rmse_per_output"][1],
            "test_rmse_accuracy": result["test_rmse_per_output"][2],
            "test_r2_stress": result["test_r2_per_output"][0],
            "test_r2_porosity": result["test_r2_per_output"][1],
            "test_r2_accuracy": result["test_r2_per_output"][2],
            "physics_residual_total": result["test_physics_residual"],
            "physics_residual_heat": result["test_physics_components"]["heat"],
            "physics_residual_stress": result["test_physics_components"]["stress"],
            "physics_residual_porosity": result["test_physics_components"]["porosity"],
            "physics_residual_geometry": result["test_physics_components"]["geometry"],
            "bound_violations_pct": result["test_bound_violations_pct"],
        }
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _print_single_result(result):
        """Print a compact summary for one variant."""
        print(f"\n--- {result['variant']} results ---")
        print(f"  Test MSE total       : {result['test_mse_total']:.6f}")
        print(f"  Test MSE per output  : stress={result['test_mse_per_output'][0]:.4f}, "
              f"porosity={result['test_mse_per_output'][1]:.6f}, "
              f"accuracy={result['test_mse_per_output'][2]:.6f}")
        print(f"  Test R2 per output   : stress={result['test_r2_per_output'][0]:.4f}, "
              f"porosity={result['test_r2_per_output'][1]:.4f}, "
              f"accuracy={result['test_r2_per_output'][2]:.4f}")
        print(f"  Physics residual     : {result['test_physics_residual']:.6f}")
        print(f"  Bound violations (%) : {result['test_bound_violations_pct']:.4f}")
        print("-" * 40 + "\n")

    def run(self):
        """Run the full ablation study and persist the results."""
        self.prepare_data()
        configs = self.build_variant_configs()

        names = [self.variant] if self.variant else list(VARIANTS)
        self.results = []
        for name in names:
            result = self.run_variant(name, configs[name])
            self.results.append(result)

        # Only write the aggregate table when all three variants have been run.
        if self.variant is None:
            self.save_table()
            self.print_table()
        return self.results

    @staticmethod
    def _fmt_float(x):
        return f"{x:.6f}"

    def save_table(self):
        """Write a CSV and Markdown summary of the ablation results."""
        csv_path = self.output_dir / "ablation_results.csv"
        md_path = self.output_dir / "ablation_results.md"

        fieldnames = [
            "variant",
            "test_mse_total",
            "test_mse_stress",
            "test_mse_porosity",
            "test_mse_accuracy",
            "test_rmse_stress",
            "test_rmse_porosity",
            "test_rmse_accuracy",
            "test_r2_stress",
            "test_r2_porosity",
            "test_r2_accuracy",
            "physics_residual_total",
            "physics_residual_heat",
            "physics_residual_stress",
            "physics_residual_porosity",
            "physics_residual_geometry",
            "bound_violations_pct",
        ]

        rows = []
        for r in self.results:
            rows.append({
                "variant": r["variant"],
                "test_mse_total": r["test_mse_total"],
                "test_mse_stress": r["test_mse_per_output"][0],
                "test_mse_porosity": r["test_mse_per_output"][1],
                "test_mse_accuracy": r["test_mse_per_output"][2],
                "test_rmse_stress": r["test_rmse_per_output"][0],
                "test_rmse_porosity": r["test_rmse_per_output"][1],
                "test_rmse_accuracy": r["test_rmse_per_output"][2],
                "test_r2_stress": r["test_r2_per_output"][0],
                "test_r2_porosity": r["test_r2_per_output"][1],
                "test_r2_accuracy": r["test_r2_per_output"][2],
                "physics_residual_total": r["test_physics_residual"],
                "physics_residual_heat": r["test_physics_components"]["heat"],
                "physics_residual_stress": r["test_physics_components"]["stress"],
                "physics_residual_porosity": r["test_physics_components"]["porosity"],
                "physics_residual_geometry": r["test_physics_components"]["geometry"],
                "bound_violations_pct": r["test_bound_violations_pct"],
            })

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Markdown summary
        lines = [
            "# PINN Ablation Results\n",
            "| Variant | MSE (Stress) | MSE (Porosity) | MSE (Accuracy) | "
            "RMSE (Stress) | RMSE (Porosity) | RMSE (Accuracy) | "
            "R² (Stress) | R² (Porosity) | R² (Accuracy) | "
            "Physics Residual | Bound Violations (%) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['variant']} | "
                f"{self._fmt_float(row['test_mse_stress'])} | "
                f"{self._fmt_float(row['test_mse_porosity'])} | "
                f"{self._fmt_float(row['test_mse_accuracy'])} | "
                f"{self._fmt_float(row['test_rmse_stress'])} | "
                f"{self._fmt_float(row['test_rmse_porosity'])} | "
                f"{self._fmt_float(row['test_rmse_accuracy'])} | "
                f"{self._fmt_float(row['test_r2_stress'])} | "
                f"{self._fmt_float(row['test_r2_porosity'])} | "
                f"{self._fmt_float(row['test_r2_accuracy'])} | "
                f"{self._fmt_float(row['physics_residual_total'])} | "
                f"{self._fmt_float(row['bound_violations_pct'])} |"
            )
        md_path.write_text("\n".join(lines) + "\n")

        print(f"\nAblation results saved to:\n  {csv_path}\n  {md_path}")

    def print_table(self):
        """Print the ablation table to stdout."""
        print("\n" + "=" * 120)
        print("ABLATION RESULTS")
        print("=" * 120)
        header = (
            f"{'Variant':<20} "
            f"{'MSE Stress':>12} {'MSE Porosity':>14} {'MSE Accuracy':>14} "
            f"{'R2 Stress':>10} {'R2 Porosity':>12} {'R2 Accuracy':>12} "
            f"{'Physics':>12} {'Violations %':>13}"
        )
        print(header)
        print("-" * len(header))
        for r in self.results:
            print(
                f"{r['variant']:<20} "
                f"{r['test_mse_per_output'][0]:>12.4f} "
                f"{r['test_mse_per_output'][1]:>14.6f} "
                f"{r['test_mse_per_output'][2]:>14.6f} "
                f"{r['test_r2_per_output'][0]:>10.4f} "
                f"{r['test_r2_per_output'][1]:>12.4f} "
                f"{r['test_r2_per_output'][2]:>12.4f} "
                f"{r['test_physics_residual']:>12.4f} "
                f"{r['test_bound_violations_pct']:>13.4f}"
            )
        print("=" * 120 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run the LPBF PINN physics-vs-bounds ablation study."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/params.yaml",
        help="Path to the base configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/optimized",
        help="Directory where the ablation table is saved.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads used by each trainer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs for the ablation.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=list(VARIANTS.keys()),
        help="Run a single ablation variant instead of the full study.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate the synthetic dataset before training.",
    )
    parser.add_argument(
        "--scan-vectors",
        type=int,
        default=50,
        help="Number of scan vectors for synthetic data generation.",
    )
    parser.add_argument(
        "--points-per-vector",
        type=int,
        default=64,
        help="Number of spatial points per scan vector.",
    )
    args = parser.parse_args()

    study = AblationStudy(
        config_path=args.config,
        output_dir=args.output_dir,
        num_threads=args.num_threads,
        seed=args.seed,
        epochs=args.epochs,
        variant=args.variant,
        regenerate=args.regenerate,
        scan_vectors=args.scan_vectors,
        points_per_vector=args.points_per_vector,
    )
    study.run()


if __name__ == "__main__":
    main()
