import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as P

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class nnBasis(nn.Module):
    def __init__(self, num_expansions):
        super(nnBasis, self).__init__()
        self.features = 256
        self.num_expansion = num_expansions
        # Common feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(1, self.features), nn.Tanh(),
            nn.Linear(self.features, self.features), nn.Tanh()
        ).to(device)
        
        # Learnable coefficient matrix (before orthogonalization)
        self.coeff_matrix = nn.Linear(self.features, self.num_expansion, bias=False).to(device)
        
    def forward(self, x):
        features = self.feature_extractor(x.view(-1, 1))  # Shared feature space
        raw_basis = self.coeff_matrix(features)  # Raw basis before orthogonalization
        orth_basis = self.gram_schmidt_process(raw_basis)  # Enforce orthogonality
        return orth_basis, self.smoothness_loss(orth_basis)

    def gram_schmidt_process(self, vectors):
        """Enforces orthogonality on output vectors using Gram-Schmidt."""
        basis = []
        for v in vectors.T:  # Iterate over each basis function (column-wise)
            w = v.clone()  # Create a clone of the vector to avoid modifying the original
            for w_ in basis:
                dot_product = torch.dot(w_, v)
                w = w - dot_product / torch.dot(w_, w_) * w_
            norm = torch.norm(w) + 1e-8  # Add a small value to avoid division by zero
            w = w/norm  # Normalize to unit norm
            basis.append(w)
        return torch.stack(basis, dim=1)  # Stack to form orthonormal basis
    
    def smoothness_loss(self, output):
        # Compute second-order differences for smoothness loss
        diff1 = output[:, 1:] - output[:, :-1]  # First-order difference
        diff2 = diff1[:, 1:] - diff1[:, :-1]  # Second-order difference
        smooth_loss = torch.mean(diff2 ** 2)
        return smooth_loss
