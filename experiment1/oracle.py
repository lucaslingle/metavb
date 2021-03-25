import numpy as np

class Oracle:
    def __init__(self, K, C):
        self.K = K
        self.C = C

    def generate_episode(self, episode_len, memory_state):
        T = episode_len
        R0 = memory_state.R
        L0 = np.linalg.cholesky(memory_state.U)

        samples = []
        epsilon_M = np.random.normal(loc=0.0, scale=1.0, size=(self.K,self.C)) # sample MN(0_KxC,I_K,I_C)
        M = R0 + np.einsum('kl,lc->kc', L0, epsilon_M) # sample p(M) = MN(M|R=R0, U=U0, V=I_C)
        for i in range(0, T):
            w = np.random.normal(loc=0.0, scale=1.0, size=(self.K,)) # sample p(w) = N_{K}(w|Mu=0_K, Sigma=I_K)
            MTw = np.einsum('kc,k->c', M, w)
            z = MTw + np.random.normal(loc=0.0, scale=1.0, size=(self.C,)) # sample p(z|w,M) = N_{C}(z|Mu=M^{T}w, Sigma=I_C)
            samples.append(z)

        samples = [np.expand_dims(z, 0) for z in samples]
        Z = np.concatenate(samples, axis=0)
        return Z

