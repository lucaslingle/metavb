import numpy as np
from utils import *

class MemoryWriterDKM:
    def __init__(self, K, C, sigma2_w):
        self.K = K
        self.C = C
        self.sigma2_w = sigma2_w # variance for q(w)

    def dynamic_addressing(self, z, memory_state):
        Lambda_w = np.einsum('kc,lc->kl', memory_state.R, memory_state.R) + np.eye(self.K)
        Eta_w = np.einsum('kc,c->k', memory_state.R, z)
        Mu_w = np.einsum('kl,l->k', np.linalg.inv(Lambda_w), Eta_w)
        Sigma_w = self.sigma2_w * np.eye(self.K)
        qw = DistributionalAddress(Mu_w=Mu_w, Sigma_w=Sigma_w)
        return qw

    def online_update_rule(self, z, w, memory_state):
        Sigma_cT = np.einsum('kl,l->k', memory_state.U, w)
        wUw = np.einsum('k,k->', w, Sigma_cT)
        Sigma_xi = 1.0
        Sigma_z = wUw + Sigma_xi
        Sigma_z_inv = 1.0 / Sigma_z
        DeltaT = z - np.einsum('k,kc->c', w, memory_state.R)

        Sigma_cT_times_Sigma_z_inv = np.einsum('k,->k', Sigma_cT, Sigma_z_inv)

        R_new = memory_state.R + np.einsum('k,c->kc', Sigma_cT_times_Sigma_z_inv, DeltaT)
        U_new = memory_state.U - np.einsum('k,l->kl', Sigma_cT_times_Sigma_z_inv, Sigma_cT)
        
        qM_new = DistributionalMemory(R=R_new, U=U_new)
        return qM_new

    def write_episode(self, Z, pM):
        T = Z.shape[0]
        memory_state = pM
        address_distributions = {}

        for t in range(0, T):
            z_t = Z[t,:]
            qw_t = self.dynamic_addressing(z=z_t, memory_state=memory_state)
            qM = self.online_update_rule(z=z_t, w=qw_t.Mu_w, memory_state=memory_state)

            address_distributions[t] = qw_t
            memory_state = qM

        return address_distributions, memory_state
