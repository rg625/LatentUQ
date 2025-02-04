import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Integrator:
    def __init__(self, time=torch.linspace(0, 1, 100), initial_value=0.0, device=device) -> None:
        self.time = time.to(device)
        self.initial_value = initial_value
        self.device = device

    def __call__(self, coeffs):
        # Calculate time intervals
        delta_t = self.time[1:] - self.time[:-1]  # [99]
        delta_t = delta_t.unsqueeze(0)  # [1, 99] for broadcasting
        
        # Store original shape
        coeffs_original_shape = coeffs.shape

        # Flatten if coeffs is 3D
        if coeffs.dim() == 3:
            coeffs = coeffs.view(-1, coeffs.shape[-1])  # Reshape to [batch_size * features, time_steps]

        # Perform cumulative sum integration
        y = torch.cumsum(coeffs[:, :-1] * delta_t, dim=1)  # [batch_size, time_steps - 1]
        # Add the initial value
        initial_column = torch.full((coeffs.size(0), 1), self.initial_value, device=self.device)  # [batch_size, 1]
        y = torch.cat([initial_column, y], dim=1)  # [batch_size, time_steps]

        # Restore original shape
        return y.view(coeffs_original_shape)