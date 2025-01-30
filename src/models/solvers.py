import torch
import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DarcySolver(nn.Module):
    def __init__(self, x, amplitude = 0.1):
        super(DarcySolver, self).__init__()
        self.k = amplitude  # Constant permeability
        self.x = x.view(-1, 1) if x.dim() < 2 else x
        self.n = len(x)
        self.device = device

    # Define basis functions and their gradients for 1D
    def basis_functions(self, x):
        return [(1 - x), x]

    def basis_gradients(self):
        return [-1, 1]

    # Define initial condition function (batched)
    def initial_condition(self, x):
        return torch.sin(torch.pi * x)  # Example: sin(pi * x)

    # Assemble the stiffness matrix and load vector for batched inputs
    def assemble_system(self, forcing_functions):
        h = 1 / (self.n - 1)
        batch_size = forcing_functions.shape[0]
        K = torch.zeros((batch_size, self.n, self.n), dtype=torch.float32).cuda()
        funcs = torch.zeros((batch_size, self.n, 1), dtype=torch.float32).cuda()
        
        # Create local stiffness matrix
        local_K = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32).cuda() * (self.k / h)
        
        # Vectorized assembly of global stiffness matrix
        for i in range(self.n - 1):
            K[:, i:i+2, i:i+2] += local_K

        # Vectorized assembly of load vector
        for a in range(2):
            funcs[:, 1:-1] += (forcing_functions[:, 1:-1] * self.basis_functions(0.5)[a] * h / 2).unsqueeze(2)

        # Apply boundary conditions (Dirichlet)
        K[:, 0, :] = 0
        K[:, :, 0] = 0
        K[:, 0, 0] = 1
        funcs[:, 0] = self.initial_condition(torch.tensor(0.0).cuda()).repeat(batch_size, 1)  # Boundary condition at x=0

        K[:, -1, :] = 0
        K[:, :, -1] = 0
        K[:, -1, -1] = 1
        funcs[:, -1] = self.initial_condition(torch.tensor(1.0).cuda()).repeat(batch_size, 1)  # Boundary condition at x=1

        return K, funcs

    # Solve the linear system for batched inputs
    def solve_system(self, K, funcs):
        return torch.linalg.solve(K, funcs)


    # Forward function to take forcing functions and return solutions
    def forward(self, forcing_functions):
        forcing_functions_shape = forcing_functions.shape

        # Flatten if coeffs is 3D
        if forcing_functions.dim() == 3:
            forcing_functions = forcing_functions.view(-1, forcing_functions.shape[-1])  # Reshape to [batch_size * features, time_steps]

        K, Funcs = self.assemble_system(forcing_functions.squeeze())
        solutions = self.solve_system(K, Funcs)
        return solutions.view(forcing_functions_shape)