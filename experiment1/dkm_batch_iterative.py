import numpy as np
from utils import *


class MemoryWriterDKMBatchIterative:
    def __init__(self, K, C, qw_sigma2):
        self.K = K
        self.C = C
        self.qw_sigma2 = qw_sigma2

    def batch_address_update_step(self, Z, qM):
        RRT_plus_IK = np.einsum('kc,lc->kl', qM.R, qM.R) + np.eye(self.K)
        RZ = np.einsum('kc,tc->tk', qM.R, Z)
        Mu_W = np.einsum('kl,tl->tk', np.linalg.inv(RRT_plus_IK), RZ)
        Sigma_w = self.qw_sigma2 * np.eye(self.K)
        qW = DistributionalAddresses(Mu_W=Mu_W, Sigma_w=Sigma_w)
        return qW

    def batch_memory_update_step(self, Z, qW, pM):
        # this is the same as the *batched* online update rule using Z = Z and W = Mu_W

        T = Z.shape[0]
        U0_inv = np.linalg.inv(pM.U)
        U0_inv_R0 = np.einsum('kl,lc->kc', U0_inv, pM.R)

        Lambda_M = U0_inv + np.einsum('tk,tl->kl', qW.Mu_W, qW.Mu_W)
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
            qW = self.batch_address_update_step(Z=Z, qM=qM)
            qM = self.batch_memory_update_step(Z=Z, qW=qW, pM=pM)

            if i == opt_iters-1:
                for t in range(0, T):
                    # here on the last iteration we save addresses separately for possible downstream use
                    qw_t = DistributionalAddress(Mu_w=qW.Mu_W[t,:], Sigma_w=qW.Sigma_w)
                    address_distributions[t] = qw_t

        return address_distributions, qM
