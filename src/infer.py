"""Real-time inference CLI for the trained LPBF PINN surrogate."""

import argparse
import json

import torch
import yaml

from pinn.model import PINN


def load_model(config, model_path, device=None):
    """Rebuild a PINN from a checkpoint.

    Args:
        config (dict): Configuration dictionary.
        model_path (str): Path to the checkpoint file.
        device (torch.device, optional): Device to load the model on.

    Returns:
        PINN: Model in evaluation mode.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint['model_state_dict']

    scaler_params = _extract_scaler_params(state)
    model_config = config['model']
    model = PINN(
        in_dim=model_config['input_dim'],
        out_dim=model_config['output_dim'],
        width=model_config['hidden_width'],
        depth=model_config['hidden_depth'],
        dropout_rate=model_config.get('dropout_rate', 0.1),
        apply_output_bounds=model_config.get('apply_output_bounds', False),
        output_bounds_temperature=model_config.get('output_bounds_temperature', 100.0),
        scaler_params=scaler_params,
        predict_temperature=model_config.get('predict_temperature', False),
    )
    model.load_state_dict(state)
    model.eval()
    return model.to(device), device


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


def build_input(param_values, input_dim, device):
    """Build a model input tensor from parameter values.

    Args:
        param_values (list[float]): Process parameter values.
        input_dim (int): Expected total input dimension.
        device (torch.device): Target device.

    Returns:
        torch.Tensor: Input tensor [1, input_dim].
    """
    n_params = len(param_values)
    if n_params + 4 != input_dim:
        raise ValueError(
            f"Parameter count ({n_params}) + 4 (coords/time) must equal "
            f"input_dim ({input_dim})"
        )
    x = torch.tensor([param_values], dtype=torch.float32, device=device)
    coords = torch.zeros(1, 3, device=device)
    time = torch.ones(1, 1, device=device)
    return torch.cat([x, coords, time], dim=1)


def predict(config, model_path, param_values, mc_dropout=False, num_samples=100):
    """Run a single inference and return a JSON-serialisable result dict."""
    model, device = load_model(config, model_path)
    model_input = build_input(param_values, config['model']['input_dim'], device)

    objective_names = config['optimizer']['objectives']

    if mc_dropout:
        mean, std = model.predict_with_uncertainty(model_input, num_samples=num_samples)
        mean = mean.detach().cpu().numpy().flatten()
        std = std.detach().cpu().numpy().flatten()
        return {
            'parameters': param_values,
            'prediction': {
                name: float(mean[i]) for i, name in enumerate(objective_names)
            },
            'uncertainty': {
                name: float(std[i]) for i, name in enumerate(objective_names)
            },
        }

    with torch.no_grad():
        pred = model(model_input).detach().cpu().numpy().flatten()
    return {
        'parameters': param_values,
        'prediction': {
            name: float(pred[i]) for i, name in enumerate(objective_names)
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run real-time inference with a trained LPBF PINN surrogate'
    )
    parser.add_argument('--config', type=str, default='data/params.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained PINN checkpoint')
    parser.add_argument('--params', type=str, required=True,
                        help='JSON object with process parameters, e.g. '
                             '{"P": 300, "v": 1000, "h": 0.1, "theta": 45, '
                             '"l_island": 5, "layer_thickness": 0.03}')
    parser.add_argument('--mc-dropout', action='store_true',
                        help='Return MC Dropout mean and uncertainty')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of MC Dropout samples')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    param_dict = json.loads(args.params)
    param_names = list(config['optimizer']['param_bounds'].keys())
    param_values = [float(param_dict[p]) for p in param_names]

    result = predict(
        config,
        args.model,
        param_values,
        mc_dropout=args.mc_dropout,
        num_samples=args.num_samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
