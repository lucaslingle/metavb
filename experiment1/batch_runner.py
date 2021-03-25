import numpy as np
import sys
import uuid
import os

from utils import DistributionalMemory
from dkm_batch_iterative import MemoryWriterDKMBatchIterative
from vbm_batch_iterative import MemoryWriterVBMBatchIterative
from evaluator import Evaluator
from oracle import Oracle
from heatmap import heatmap_saver

class Runner:
    def __init__(self, T, K, C, sigma2_w, opt_iters, R0_init_scale):
        self.T = T
        self.K = K
        self.C = C
        self.sigma2_w = sigma2_w
        self.opt_iters = opt_iters
        self.R0_init_scale = R0_init_scale

        self.evaluator = Evaluator(K=self.K, C=self.C)
        self.DKM = MemoryWriterDKMBatchIterative(K=self.K, C=self.C, qw_sigma2=self.sigma2_w)
        self.VBM = MemoryWriterVBMBatchIterative(K=self.K, C=self.C)

        self.R0 = np.random.normal(loc=0.0, scale=self.R0_init_scale, size=(self.K,self.C))
        self.U0 = np.eye(K)

        self.pM = DistributionalMemory(R=self.R0, U=self.U0)

        self.orcl = Oracle(K=K, C=C)

    def run(self, with_info=False):
        Z = self.orcl.generate_episode(episode_len=self.T, memory_state=self.pM)
        DKM_qW, DKM_qM = self.DKM.write_episode(Z=Z, pM=self.pM, opt_iters=self.opt_iters)
        VBM_qW, VBM_qM = self.VBM.write_episode(Z=Z, pM=self.pM, opt_iters=self.opt_iters)

        DKM_elbo_per_frame = self.evaluator.compute_elbo_per_frame(Z=Z, qW=DKM_qW, qM=DKM_qM, pM=self.pM)
        VBM_elbo_per_frame = self.evaluator.compute_elbo_per_frame(Z=Z, qW=VBM_qW, qM=VBM_qM, pM=self.pM)

        info = {}
        if with_info:
            DKM_tr_Uf = np.sum(np.diag(DKM_qM.U))
            VBM_tr_Uf = np.sum(np.diag(VBM_qM.U))
            info['DKM_tr_Uf'] = DKM_tr_Uf
            info['VBM_tr_Uf'] = VBM_tr_Uf

        return DKM_elbo_per_frame, VBM_elbo_per_frame, info


if __name__ == '__main__':
    mode = str(sys.argv[1])
    print('running...')

    if mode == 'simple':
        r = Runner(T=1, K=32, C=200, sigma2_w=0.3, R0_init_scale=0.05, opt_iters=50)
        DKM_elbo_per_frame, VBM_elbo_per_frame, _ = r.run()
        print('DKM_elbo_per_frame: {}'.format(DKM_elbo_per_frame))
        print('VBM_elbo_per_frame: {}'.format(VBM_elbo_per_frame)) 

    else:
        T = int(str(sys.argv[1]))
        opt_iters = int(str(sys.argv[2]))
        Ks = [8, 16, 32, 64, 128, 256]
        Cs = [50, 100, 200, 400, 800][::-1]
        sigma2_w = 0.3
        R0_init_scale = 1.0
        ratios = np.zeros(dtype=np.float32, shape=(len(Cs), len(Ks)))
        for j in range(0,len(Ks)):
            K = Ks[j]
            for i in range(0,len(Cs)):
                C = Cs[i]
                r = Runner(T=T, K=K, C=C, sigma2_w=sigma2_w, R0_init_scale=R0_init_scale, opt_iters=opt_iters)
                DKM_elbo_per_frame, VBM_elbo_per_frame, _ = r.run()
                assert DKM_elbo_per_frame < 0.0 and VBM_elbo_per_frame < 0.0
                ratios[i,j] = DKM_elbo_per_frame / VBM_elbo_per_frame # since both are negative due to sigma_z2 = 1, this fraction signifies how much better vbm algo is.

        R0scale_int, R0scale_decimal = str(R0_init_scale).split(".")
        id = str(uuid.uuid4())
        fn = 'heatmap_batchwrite_elboratio_kc_t{}_o{}_R0initscale{}pt{}_id{}.png'.format(T, opt_iters, R0scale_int, R0scale_decimal, id)
        fp = os.path.join('output/', fn)

        heatmap_saver(
            x_names=Ks, y_names=Cs, arr=ratios, fp=fp, title='outperformance ratio')

        print('saved as {}'.format(fp))
