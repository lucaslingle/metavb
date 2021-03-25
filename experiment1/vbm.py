import numpy as np
from utils import *


class MemoryWriterVBM:
    def __init__(self, K, C):
        self.K = K
        self.C = C

    def vb_e_step(self, Z, qM):
        Lambda_w = np.einsum('kc,lc->kl', qM.R, qM.R) + np.eye(self.K) + float(self.C) * qM.U
        Eta_w = np.einsum('kc,tc->tk', qM.R, Z)
        Sigma_w = np.linalg.inv(Lambda_w)
        Mu_W = np.einsum('kl,tl->tk', Sigma_w, Eta_w)
        qW = DistributionalAddresses(Mu_W=Mu_W, Sigma_w=Sigma_w)
        return qW

    def vb_m_step(self, Z, qW, pM):
        T = Z.shape[0]
        U0_inv = np.linalg.inv(pM.U)
        U0_inv_R0 = np.einsum('kl,lc->kc', U0_inv, pM.R)

        Lambda_M = U0_inv + np.einsum('tk,tl->kl', qW.Mu_W, qW.Mu_W) + float(T) * qW.Sigma_w
        Eta_M = U0_inv_R0 + np.einsum('tk,tc->kc', qW.Mu_W, Z)

        U_new = np.linalg.inv(Lambda_M)
        R_new = np.einsum('kl,lc->kc', U_new, Eta_M)
        
        qM_new = DistributionalMemory(R=R_new, U=U_new)
        return qM_new

    def write_episode(self, Z, pM, opt_iters=1):
        T = Z.shape[0]
        qM = pM
        address_distributions = {}

        for i in range(0, opt_iters):
            qW = self.vb_e_step(Z=Z, qM=qM)
            qM = self.vb_m_step(Z=Z, qW=qW, pM=pM)

            if i == opt_iters-1:
                for t in range(0, T):
                    qw_t = DistributionalAddress(Mu_w=qW.Mu_W[t,:], Sigma_w=qW.Sigma_w)
                    address_distributions[t] = qw_t

        return address_distributions, qM
