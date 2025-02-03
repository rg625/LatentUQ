import torch
import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define the Branch Network
class BranchNet(nn.Module):
    def __init__(self, latent_dim, branch_out_dim):
        super(BranchNet, self).__init__()
        self.gen_feature = 32
        self.brach = nn.Sequential(nn.Linear(latent_dim, self.gen_feature),
                                  nn.Softplus(),
                                  nn.Linear(self.gen_feature, self.gen_feature),
                                  nn.Softplus(),
                                  nn.Linear(self.gen_feature, branch_out_dim)).to(device)

    def forward(self, z):
        return self.brach(z)

# Define the Trunk Network
class TrunkNet(nn.Module):
    def __init__(self, trunk_out_dim):
        super(TrunkNet, self).__init__()
        self.gen_feature = 32
        self.trunk = nn.Sequential(nn.Linear(1, self.gen_feature),
                                  nn.Softplus(),
                                  nn.Linear(self.gen_feature, self.gen_feature),
                                  nn.Softplus(),
                                  nn.Linear(self.gen_feature, trunk_out_dim)).to(device)

    def forward(self, t):
        # Reshape t to [batch_size, num_points, 1] for processing
        if t.dim()<2:
            t  = t.view(-1, 1)  # Add channel dimension if needed
        return self.trunk(t)

# Define the combined DeepONet-like decoder
class DeepONet(nn.Module):
    def __init__(self, latent_dim, output_dim = 1, branch_out_dim=8, trunk_out_dim=4):
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet(latent_dim, branch_out_dim)
        self.trunk_net = TrunkNet(trunk_out_dim)
        self.fc = nn.Linear(branch_out_dim * trunk_out_dim, output_dim)  # Output single function value per point

    def forward(self, indices, t):
        """
        Inputs:
        - z: Latent vector [batch_size, latent_dim]
        - t: Time points [batch_size, num_points]

        Output:
        - Function values at specified time points [batch_size, num_points]
        """
        # Pass the latent vector through the branch network
        branch_out = self.branch_net(indices)  # Shape: [branch_out_dim]
        # Pass the time points through the trunk network
        trunk_out = self.trunk_net(t)#.unsqueeze(0).expand(z.size(0), -1, -1)  # Shape: [num_points, trunk_out_dim]
        
        # Compute the outer product and flatten for the linear layer
        combined = torch.einsum('nT,B->nTB', trunk_out, branch_out)  # Shape: [num_points, trunk_out_dim, branch_out_dim]
        combined = combined.view(combined.size(0), -1)  # Flatten trunk_out_dim * branch_out_dim
        # Raw output before enforcing orthogonality
        y_raw = self.fc(combined)  # Shape: [num_points, N]

        # Apply QR decomposition across time dimension
        Q, _ = torch.linalg.qr(y_raw)  # Q is orthogonal
        return Q  # Shape: [time, N]