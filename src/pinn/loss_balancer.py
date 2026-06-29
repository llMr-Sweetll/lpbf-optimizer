import torch


class AdaptiveLossBalancer:
    """
    Adaptive Loss Balancing for Physics-Informed Neural Networks.

    This class implements a dynamic weight adjustment mechanism inspired by GradNorm
    and Gradient Pathology analysis to balance the training signals from different
    loss components (Data, Heat, Stress).

    References:
        [1] Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating
            gradient flow pathologies in physics-informed neural networks.
            SIAM Journal on Scientific Computing, 43(5), A3055-A3081.

        [2] Chen, Z., et al. (2018). GradNorm. ICML.

        [3] "Advances in PINNs for Additive Manufacturing" (2024).
            Discusses importance of adaptive weighting for multi-physics AM problems.
    """

    def __init__(self, num_losses, alpha=1.5):
        """
        Initialize the loss balancer.

        Args:
            num_losses (int): Number of loss components to balance.
            alpha (float): Hyperparameter controlling the strength of the restoring force.
                           Higher alpha pulls weights back to 1.0 more strongly.
        """
        self.num_losses = num_losses
        self.alpha = alpha

        # Initialize weights to 1.0
        # These are learnable parameters or updated manually
        self.weights = torch.ones(num_losses, dtype=torch.float32, requires_grad=False)

        # To smooth the weight updates
        self.initial_losses = None

    def update_weights(self, losses, shared_layer_params):
        """
        Update loss weights based on gradient magnitudes.

        Args:
            losses (list of torch.Tensor): List of individual loss components (unweighted).
            shared_layer_params (torch.Tensor/Parameter): Parameters of the last shared layer
                                                        to compute gradients with respect to.

        Returns:
            torch.Tensor: Normalized weights for the current step.
        """
        if self.initial_losses is None:
            self.initial_losses = torch.tensor([loss.item() for loss in losses], device=losses[0].device)

        # We need to compute gradients of each loss w.r.t shared parameters
        # Note: This adds computational overhead (multiple backward passes)
        # For efficiency in PINNs, we often approximate or use just the gradient norms directly
        # if available. Here we assume we can run a separate backward for re-weighting.

        grads = []
        for loss in losses:
            # Differentiate loss w.r.t shared layer weights
            # retain_graph=True because we need the graph for the actual optimization step later
            g = torch.autograd.grad(loss, shared_layer_params, retain_graph=True, create_graph=False)[0]
            grads.append(torch.norm(g, 2))

        grad_norms = torch.stack(grads)

        # Calculate the average gradient norm
        avg_grad_norm = torch.mean(grad_norms)

        # Calculate the relative inverse training rate for each task
        # Inverse training rate r_i(t) = L_i(t) / L_i(0)
        current_losses = torch.tensor([loss.item() for loss in losses], device=losses[0].device)
        relative_losses = current_losses / (self.initial_losses + 1e-8)
        avg_relative_loss = torch.mean(relative_losses)

        inverse_training_rates = relative_losses / (avg_relative_loss + 1e-8)

        # The target gradient norm for each task is avg_grad_norm * (inverse_training_rate)^alpha
        target_grad_norms = avg_grad_norm * (inverse_training_rates ** self.alpha)

        # In standard GradNorm, we would optimize a separate loss L_grad to update weights.
        # Here, for stability and simplicity in this engineering context,
        # we update weights directly to bring gradients closer to targets.

        # If grad_norm_i < target_i, we want to increase weight w_i
        # w_i_new = w_i * (target_i / grad_norm_i)

        new_weights = self.weights.to(losses[0].device) * (target_grad_norms / (grad_norms + 1e-8))

        # Normalize weights so they sum to num_losses (keep scale consistent)
        new_weights = new_weights / new_weights.sum() * self.num_losses

        # Update internal weights with moving average for stability
        beta = 0.9  # Smoothing factor
        self.weights = beta * self.weights.to(losses[0].device) + (1 - beta) * new_weights.detach()

        return self.weights
