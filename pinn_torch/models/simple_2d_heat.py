import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
"""
참고: http://ramanujan.math.trinity.edu/rdaileda/teach/s15/m3357/lectures/lecture_3_5_slides.pdf
u_t=u_xx+u_yy 형태의 heat equation을 풀기 위한 네트워크

rectangular grid R = [0, a] X [0, b]

Boundary conditions
Initial temperature: 
- u(x, y, 0)=f(x, y), (x, y) in R

Homogeneous Dirichlet:
- u(0, y, t)=u(a, y, t)=0, 0 <= y <= b, t > 0
- u(x, 0, t)=u(x, b, t)=0, 0 <= x <= a, t > 0

해당 방정식의 해
"""
class Net(nn.Module):
    def __init__(self, hidden_units: int = 5, hidden_depth: int = 5, bias: bool = True, activation: nn.modules.activation = nn.Sigmoid()):
        super(Net, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(3, hidden_units, bias = bias), # x, y, t 
            activation
        )
        
        layers = []
        for _ in range(hidden_depth):
            layers.append(nn.Linear(hidden_units, hidden_units, bias = bias))
            layers.append(activation)

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_units, 1, bias = bias)

    
    def forward(self, x, y, t):
        input_vector = torch.cat([x, y, t], dim=1)
        input_layer_output = self.input_layer(input_vector)
        hidden_layer_output = self.hidden_layers(input_layer_output)
        output = self.output_layer(hidden_layer_output)
        return output
    
    def pde(self, x, y, t):
        u = self(x, y, t)
        u_x = grad(u.sum(), x, create_graph=True)[0]
        u_xx = grad(u_x.sum(), x, create_graph=True)[0]
        u_y = grad(u.sum(), y, create_graph=True)[0]
        u_yy = grad(u_y.sum(), y, create_graph=True)[0]
        u_t = grad(u.sum(), t, create_graph=True)[0]
        pde = u_t - u_xx - u_yy
        return pde
        
    def analytic_sol(self, a, b, c, f, x, y, t):
        # TODO
        mu_m = lambda m: m * np.pi / a
        nu_n = lambda n: n * np.pi / b
        lambda_mn = lambda m, n: c * np.sqrt(mu_m(m) ** 2 + nu_n(n) ** 2)
        A_mn = 4/(a*b) 
        pass



    
    

