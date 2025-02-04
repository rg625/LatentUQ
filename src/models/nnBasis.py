import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as P

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class nnBasis(nn.Module):
    def __init__(self, num_expansions):
        super(nnBasis, self).__init__()
        self.features = 64
        self.num_expansion = num_expansions

        self.coeff_matrix = nn.Sequential(
            nn.Linear(1, self.features, bias=True), nn.Tanh(),
            nn.Linear(self.features, self.features, bias=False), nn.Tanh(),
            nn.Linear(self.features, self.features, bias=False), nn.Tanh(),
            nn.Linear(self.features, self.features, bias=False), nn.Tanh(),
            nn.Linear(self.features, self.features, bias=False), nn.Tanh(),
            nn.Linear(self.features, self.num_expansion, bias=False), 
        ).to(device)
    
    def forward(self, time):
        raw_basis = self.coeff_matrix(time.view(-1, 1))  # Raw basis before orthogonalization
        orth_basis = self.orthogonalize(raw_basis)  # Enforce orthogonality
        return orth_basis
    
    def orthogonalize(self, W):
        Q, R = torch.linalg.qr(W)  # QR decomposition
        return Q  # Ensures W lies on the Stiefel manifold
    
    def smoothness_loss(self, output):
        # Compute second-order differences for smoothness loss
        diff1 = output[:, 1:] - output[:, :-1]  # First-order difference
        diff2 = diff1[:, 1:] - diff1[:, :-1]  # Second-order difference
        smooth_loss = torch.mean(diff2 ** 2) + torch.mean(diff1 ** 2)
        return 100*smooth_loss