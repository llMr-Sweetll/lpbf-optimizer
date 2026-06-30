import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# For Bayesian optimization with Ax/BoTorch
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig


class BayesianOptimizer:
    """
    Bayesian optimization for LPBF process parameter tuning using the PINN surrogate model.

    This optimizer uses Ax/BoTorch which implements Bayesian Optimization with
    Thompson sampling. It's useful for more efficiently exploring the parameter space
    compared to genetic algorithms when the number of evaluations is limited.
    """

    def __init__(self, config_path, model_path):
        """
        Initialize the Bayesian optimizer

        Args:
            config_path: Path to the configuration file
            model_path: Path to the trained PINN model checkpoint
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the trained PINN model
        self.model = self._load_model(model_path)

        # Output directory
        self.output_dir = Path(self.config['optimizer']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self, model_path):
        """
        Load the trained PINN model

        Args:
            model_path: Path to model checkpoint

        Returns:
            Loaded PINN model
        """
        from pinn.model import PINN

        # Load the checkpoint first so that optional scaler parameters and bound
        # settings can be reconstructed before the model is instantiated.
        checkpoint = torch.load(model_path, map_location=self.device)
        state = checkpoint['model_state_dict']

        scaler_params = self._extract_scaler_params(state)

        # Create model with same architecture and settings as during training
        model_config = self.config['model']
        model = PINN(
            in_dim=model_config['input_dim'],
            out_dim=model_config['output_dim'],
            width=model_config['hidden_width'],
            depth=model_config['hidden_depth'],
            dropout_rate=model_config.get('dropout_rate', 0.1),
            apply_output_bounds=model_config.get('apply_output_bounds', False),
            output_bounds_temperature=model_config.get('output_bounds_temperature', 100.0),
            scaler_params=scaler_params,
        )

        model.load_state_dict(state)

        # Set to evaluation mode
        model.eval()
        model = model.to(self.device)

        return model

    @staticmethod
    def _extract_scaler_params(state):
        """Build the scaler_params dict from a model state_dict, if present."""
        required = ('scan_mean', 'scan_std', 'coord_mean', 'coord_std')
        if not all(k in state for k in required):
            return None
        return {
            'scan_vectors': {
                'mean': state['scan_mean'].cpu().numpy(),
                'std': state['scan_std'].cpu().numpy(),
            },
            'coordinates': {
                'mean': state['coord_mean'].cpu().numpy(),
                'std': state['coord_std'].cpu().numpy(),
            },
        }

    def _create_evaluation_function(self, objective_idx=0):
        """
        Create an evaluation function for the objective

        Args:
            objective_idx: Index of the objective to optimize (default: 0 for residual stress)

        Returns:
            Evaluation function for Ax
        """
        # Determine objective direction. BayesOpt always minimizes, so we negate
        # objectives that should be maximized (e.g., geometric_accuracy).
        objective_name = self.config['optimizer']['objectives'][objective_idx]
        maximize = objective_name == 'geometric_accuracy'

        # Define the evaluation function
        def evaluate(parameters):
            # Convert parameters to tensor
            param_values = [parameters[p] for p in self.param_names]
            x = torch.tensor([param_values], dtype=torch.float32, device=self.device)

            # Add dummy spatial coordinates and time
            batch_size = x.shape[0]
            coords = torch.zeros(batch_size, 3, device=self.device)  # Origin point
            time = torch.ones(batch_size, 1, device=self.device)     # Final time step

            # Forward pass through the model
            with torch.no_grad():
                model_input = torch.cat([x, coords, time], dim=1)
                predictions = self.model(model_input)

            # Extract target objective value
            obj_value = predictions[0, objective_idx].item()
            if maximize:
                obj_value = -obj_value

            # For stochastic models, we could return SEM, but our surrogate is deterministic
            return {"objective": (obj_value, 0.0)}

        return evaluate

    def _create_params(self):
        """
        Create range-parameter configs for the Ax Client API.

        Returns:
            list[RangeParameterConfig]: Search-space parameters.
        """
        # Get parameter bounds
        param_bounds = self.config['optimizer']['param_bounds']
        self.param_names = list(param_bounds.keys())

        # Define parameters using the new Ax config API
        parameters = []
        for name in self.param_names:
            lo, hi = param_bounds[name]
            parameters.append(
                RangeParameterConfig(name=name, bounds=(lo, hi), parameter_type="float")
            )

        return parameters

    def run_single_objective_optimization(self, objective_idx=0, n_trials=50):
        """
        Run Bayesian optimization for a single objective using the Ax Client API.

        Args:
            objective_idx: Index of the objective to optimize
            n_trials: Number of trials to run

        Returns:
            tuple: (best_parameters, best_metric_values, client)
        """
        parameters = self._create_params()

        # Get the objective name
        objective_name = self.config['optimizer']['objectives'][objective_idx]
        print(f"Optimizing for {objective_name}")

        # Ax always minimizes. For maximized objectives (geometric_accuracy)
        # we optimize the negated metric.
        maximize = objective_name == 'geometric_accuracy'
        objective_str = "-objective" if maximize else "objective"

        # Create evaluation function
        evaluation_function = self._create_evaluation_function(objective_idx)

        # Configure the Ax Client experiment
        client = Client()
        client.configure_experiment(
            parameters=parameters,
            name="lpbf_bayesopt",
        )
        client.configure_optimization(objective=objective_str)

        # Run optimization
        print(f"Running Bayesian optimization with {n_trials} trials...")
        start_time = time.time()

        for _ in range(n_trials):
            trial_params = client.get_next_trials(max_trials=1)
            for trial_index, params in trial_params.items():
                raw_data = evaluation_function(params)
                client.complete_trial(trial_index, raw_data=raw_data)

        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")

        # Retrieve best in-sample parameters
        best_parameters, best_metric_values, _, _ = client.get_best_parameterization()

        # Restore the original sign for maximized objectives when reporting.
        # ``best_metric_values`` maps metric names to (mean, sem) tuples.
        best_mean, best_sem = best_metric_values['objective']
        best_objective_display = -best_mean if maximize else best_mean

        # Print results
        print("\nBest parameters:")
        for param, value in best_parameters.items():
            print(f"  {param}: {value}")
        print(f"Best objective value: {best_objective_display} (sem={best_sem})")

        # Save results
        self._save_results(client._experiment, objective_name)

        return best_parameters, best_metric_values, client

    def run_multi_objective_optimization(self, n_trials=50):
        """
        Run a series of single-objective optimizations for each objective

        This is a simple approach to multi-objective optimization with Bayesian methods.
        For more sophisticated approaches, MOO extensions to BoTorch could be used.

        Args:
            n_trials: Number of trials per objective

        Returns:
            List of (best_parameters, best_metric_values, client) tuples.
        """
        objectives = self.config['optimizer']['objectives']
        results = []

        for i, obj_name in enumerate(objectives):
            print(f"\nOptimizing for objective {i+1}/{len(objectives)}: {obj_name}")
            result = self.run_single_objective_optimization(i, n_trials)
            results.append(result)

        return results

    def _save_results(self, experiment, objective_name):
        """
        Save optimization results

        Args:
            experiment: Ax experiment
            objective_name: Name of the objective
        """
        # Get the data from the experiment
        data = experiment.fetch_data()
        df = data.df

        # Save raw data to CSV
        csv_path = self.output_dir / f"bayesopt_{objective_name}.csv"
        df.to_csv(csv_path, index=False)

        # Extract parameters and objective values. Ax's dataframe format changed,
        # so we read arm parameters directly from the experiment trials.
        maximize = objective_name == 'geometric_accuracy'
        parameters = []
        objective_values = []

        for _, row in df.iterrows():
            trial_index = int(row["trial_index"])
            arm = experiment.trials[trial_index].arm
            params = [arm.parameters[p] for p in self.param_names]
            parameters.append(params)
            objective_values.append(row["mean"])

        # Convert to numpy arrays
        X = np.array(parameters)
        Y = np.array(objective_values).reshape(-1, 1)

        # For maximized objectives, the optimizer worked on the negated value;
        # restore the original sign for reporting.
        if maximize:
            Y = -Y

        # Save to HDF5
        with h5py.File(self.output_dir / f"bayesopt_{objective_name}.h5", 'w') as f:
            # Save process parameters
            f.create_dataset('parameters', data=X)
            # Save corresponding objective values
            f.create_dataset('objectives', data=Y)
            # Save parameter names
            dt = h5py.special_dtype(vlen=str)
            param_names = np.array(self.param_names, dtype=object)
            f.create_dataset('parameter_names', data=param_names, dtype=dt)
            # Save objective name
            obj_name = np.array([objective_name], dtype=object)
            f.create_dataset('objective_names', data=obj_name, dtype=dt)

            # Store metadata
            f.attrs['n_trials'] = len(X)
            f.attrs['n_parameters'] = X.shape[1]
            best_objective = Y.max() if maximize else Y.min()
            f.attrs['best_objective'] = best_objective

            # Store best parameters
            best_idx = Y.argmax() if maximize else Y.argmin()
            f.attrs['best_params'] = X[best_idx]

        # Generate plots
        self._plot_convergence(df, objective_name, maximize=maximize)

    def _plot_convergence(self, df, objective_name, maximize=False):
        """
        Plot optimization convergence

        Args:
            df: DataFrame with optimization data
            objective_name: Name of the objective
            maximize: Whether the objective is maximized (e.g., geometric_accuracy).
        """
        plt.figure(figsize=(10, 6))
        objective = df['mean'].values
        if maximize:
            # The optimizer minimized -objective, so restore the original metric.
            objective = -objective
            best_so_far = np.maximum.accumulate(objective)
        else:
            best_so_far = np.minimum.accumulate(objective)
        plt.plot(range(1, len(best_so_far) + 1), best_so_far)
        plt.xlabel('Trial')
        plt.ylabel(f'Best {objective_name} so far')
        plt.title('Optimization Convergence')
        plt.grid(True)
        plt.savefig(self.output_dir / f"bayesopt_convergence_{objective_name}.png")
        plt.close()

def main():
    """Main function to run Bayesian optimization"""
    import argparse

    parser = argparse.ArgumentParser(description='Run Bayesian optimization with PINN surrogate')
    parser.add_argument('--config', type=str, default='data/params.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained PINN model checkpoint')
    parser.add_argument('--objective', type=int, default=0,
                        help='Index of the objective to optimize (default: 0 for residual stress)')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of optimization trials')
    parser.add_argument('--multi', action='store_true',
                        help='Run multi-objective optimization (sequential)')

    args = parser.parse_args()

    # Create optimizer
    optimizer = BayesianOptimizer(args.config, args.model)

    # Run optimization
    if args.multi:
        optimizer.run_multi_objective_optimization(args.trials)
    else:
        optimizer.run_single_objective_optimization(args.objective, args.trials)

    # Print summary
    print("\nOptimization complete!")
    print(f"Results saved to: {optimizer.output_dir}")


if __name__ == '__main__':
    main()
