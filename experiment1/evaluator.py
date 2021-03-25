import numpy as np

class Evaluator:
    def __init__(self, K, C):
        self.K = K
        self.C = C

    @staticmethod
    def logdet(A):
        slogdet = np.linalg.slogdet(A)
        logdet_A = slogdet[0] * slogdet[1]
        return logdet_A

    @staticmethod
    def trace(A):
        tr_A = np.sum(np.diag(A))
        return tr_A

    @staticmethod
    def inv(A):
        A_inv = np.linalg.inv(A)
        return A_inv

    def compute_dkl_w(self, qw):
        ## Compute D_{KL}(q(w)||p(w)).
        ## This method uses a formula which requires that p(w) = N(0,I_K).

        # formula below uses the fact that p(w) cov mat is I_K, and the mean is O_K.
        # term1: the multiplication by Sigma_w_prior^{-1}, inside the trace, is elided.
        # term2: the vector subtraction by Mu_w_prior, inside the terms of the quadratic form, is elided. 
        # term2: the multiplication by Sigma_w_prior^{-1}, inside the quadratic form, is elided.
        # term4: the addition of logdet(Sigma_w_prior), which equals zero when Sigma_w_prior=I_K, is elided.

        term1 = self.trace(qw.Sigma_w)
        term2 = np.einsum('k,k->', qw.Mu_w, qw.Mu_w)
        term3 = -float(self.K)
        term4 = -self.logdet(qw.Sigma_w)

        dkl_w = 0.5 * (term1 + term2 + term3 + term4)
        return dkl_w

    def compute_dkl_M(self, qM, pM):
        ## Compute D_{KL}(q(M)||p(M)).
        ## This method does not assume that U_0 = I_K or that R_0 = O_{KxC}. It is general.
        ## It still assumes that V_0 and V_final are I_{C}, as per the formula from our paper (Appendix D).
        
        U0_inv = self.inv(pM.U)
        R_diff = qM.R - pM.R

        term1 = float(self.C) * self.trace(np.einsum('kl,lp->kp', U0_inv, qM.U))
        term2 = self.trace(np.einsum('kc,kz->cz', R_diff, np.einsum('kl,lc->kc', U0_inv, R_diff)))
        term3 = -float(self.K * self.C)

        term4_a = -float(self.C) * self.logdet(qM.U)
        term4_b = float(self.C) * self.logdet(pM.U)
        term4 = term4_a + term4_b

        dkl_M = 0.5 * (term1 + term2 + term3 + term4)
        return dkl_M

    def compute_expected_conditional_likelihood(self, z, qw, qM):
        ## Compute E_{w ~ q(w), M ~ q(M)}[logp(z|w,M)].
        ## We have to include derivation for this in the paper still
        # note that like the rest of the tests, we assume the memory readout noise sigma_z is 1.0 here, 
        # i.e., p(z|w,M) = N_{C}(z|mu=M^{T}w, Sigma=I_C).

        R = qM.R
        U = qM.U
        Mu_w = qw.Mu_w
        Sigma_w = qw.Sigma_w

        RRT_plus_CU = np.einsum('kc,lc->kl', R, R) + float(self.C) * U            

        Elogpz_term1 = -0.5 * (
            np.einsum('c,c->', z, z) \
            - 2.0 * np.einsum('c,c->', z, np.einsum('kc,k->c', R, Mu_w)) \
            + self.trace(np.einsum('kl,lp->kp', RRT_plus_CU, Sigma_w)) \
            + np.einsum('k,k->', Mu_w, np.einsum('kl,l->k', RRT_plus_CU, Mu_w))
        )
        Elogpz_term2 = -(float(self.C) / 2.0) * np.log(2 * np.pi)
        Elogpz = Elogpz_term1 + Elogpz_term2

        return Elogpz

    def compute_elbo_contrib_for_timestep(self, z, qw, qM):
        Elogpz = self.compute_expected_conditional_likelihood(z, qw, qM)
        dkl_w = self.compute_dkl_w(qw)
        contrib = Elogpz - dkl_w
        return contrib

    def compute_elbo_per_frame(self, Z, qW, qM, pM):
        # this method assumes:
        # Z is a numpy array of shape [T, C], 
        # qW is a distionary whose keys are integers 0, ..., T-1 and whose values are collection.namedtuple's 
        #    with fields 'Mu_w', 'Sigma_w' which are numpy arrays with shapes (K,) and (K,K) respectively, with Sigma_w positive-definite symmetric.
        # qM is a collection.namedtuple with fields 'R', 'U' which are numpy arrays with shapes [K,C] and [K,K] respectively, 
        #    and U is positive-definite and symmetric.
        # pM is a collection.namedtuple with fields 'R', 'U' which are numpy arrays with shapes [K,C] and [K,K] respectively, 
        #    and U is positive-definite and symmetric.
        episode_len = Z.shape[0]
        elbo = 0.0
        for t in range(0, episode_len):
            z = Z[t,:]
            qw = qW[t]
            elbo_contrib_t = self.compute_elbo_contrib_for_timestep(z, qw, qM)
            elbo += elbo_contrib_t

        dkl_M = self.compute_dkl_M(qM, pM)
        elbo = elbo - dkl_M

        elbo_per_frame = elbo / float(episode_len)
        return elbo_per_frame
