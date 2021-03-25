import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import collections

MemoryState = collections.namedtuple('MemoryState', field_names=['R', 'U', 'Mu_b', 'Sigma_b'])

class VariationalBayesianMemory:
    def __init__(self, hps, name=None):
        self.img_height = hps.img_height
        self.img_width = hps.img_width
        self.img_channels = hps.img_channels
        self.discrete_outputs = hps.discrete_outputs

        self.batch_size = hps.batch_size
        self.episode_len = hps.episode_len

        self.enc_blocks = hps.enc_blocks
        self.dec_blocks = hps.dec_blocks
        self.num_filters = hps.num_filters
        self.res_blocks_per_encoder_block = 1
        self.res_blocks_per_decoder_block = 1
        self.res_block_internal_dropout_rate = 0.0

        self.code_size = hps.code_size
        self.memory_size = hps.memory_size
        self.num_clusters = hps.num_clusters
        self.num_hops = hps.num_hops
        assert self.code_size % self.num_hops == 0
        self.opt_iters = hps.opt_iters

        self.sr_alpha = hps.sr_alpha
        self.sr_beta = hps.sr_beta
        self.sr_gamma = hps.sr_gamma
        self.sr_delta = hps.sr_delta
        self.sr_epsilon = hps.sr_epsilon

        self.activation = lambda vec: tf.nn.swish(vec) # this is swish-1, see tf docs

        self.lr = hps.lr

        self.global_step = tf.train.get_or_create_global_step()
        self._name = 'VBM_Tree' if name is None else name

        with tf.variable_scope(self._name):

            # placeholders used during training and evaluation.
            self.input_xs = tf.placeholder(dtype=tf.float32, shape=[None, self.episode_len, self.img_height, self.img_width, self.img_channels])
            self.is_train = tf.placeholder(dtype=tf.bool, shape=[])
            batch_size = tf.shape(self.input_xs)[0]

            # constants.
            self.v1 = 0.40
            self.v2 = 0.40
            self.pM_sigma2 = self.memory_size * self.v1
            self.pb_sigma2 = self.memory_size * self.v2
            self.pz_sigma2 = self.memory_size * (1.0 - self.v1 - self.v2)
            self.pz_logsigma = 0.5 * tf.log(self.pz_sigma2)
            self.pb_logsigma = 0.5 * tf.log(self.pb_sigma2)

            # get memory prior p(M_1:H)p(b_1:H) = prod_h=1^H p(M_h)p(b_h).
            self.pMb = self.get_memory_prior(batch_size)

            self.R0s = self.pMb.R # [b,g,h,k,c/g].
            self.U0s = self.pMb.U # [b,g,h,k,k].
            self.U0_invs = tf.linalg.inv(self.U0s) # [b,g,h,k,k].
            self.U0_inv_R0s = tf.einsum('bghkl,bghlc->bghkc', self.U0_invs, self.R0s) # [b,g,h,k,c/g].
            self.sigma2 = tf.exp(2.0 * self.pz_logsigma) # []

            self.Mu_b0s = self.pMb.Mu_b # [b,g,h,c/g].
            self.Sigma_b0s = self.pMb.Sigma_b # [b,g,h,c/g,c/g].
            self.Sigma_b0_invs = tf.linalg.inv(self.Sigma_b0s) # [b,g,h,c/g,c/g].
            self.Sigma_b0_inv_Mu_b0s = tf.einsum('bghcz,bghz->bghc', self.Sigma_b0_invs, self.Mu_b0s) # [b,g,h,c/g].

            # encode
            self.input_xs_batched = tf.reshape(self.input_xs, [-1, self.img_height, self.img_width, self.img_channels])
            self.qz_batched = self.qz(self.input_xs_batched, training=self.is_train)
            self.qZ = tfp.distributions.BatchReshape(self.qz_batched, batch_shape=[-1, self.episode_len]) # [b,t], [c].
            self.Mu_Z = self.qZ.mean()
            self.Scale_diag_Z = self.qZ.stddev()

            ## write to memory using mean-field variational Bayes
            # get a good init for episode-level latents q(M_1:H^(g))q(b_1:H^(g)), g=1,2,...,G.
            self.Z_for_init = self.Mu_Z
            self.Z_for_init_g12 = tf.reshape(self.Z_for_init, [-1, self.episode_len, self.num_hops, (self.code_size // self.num_hops)]) # [b,t,g,c/g].

            self.qMb_init_idxs_g12 = self.compute_good_init_tf_tree(self.Z_for_init_g12) # shape [b,g,h,t], batch containing onehots for initting q(b_1:H^(g)) for g=1,2.

            '''
            # lets add some ad-hoc post-training code added to test if the model works better when the cluster loc init decisions are synchronized across code segments
            # e.g., use full codes for making k-means++ init decisions, and assign segments to components according to this full-code clustering init.
            self.qMb_init_idxs = self.compute_good_init_tf(self.Z_for_init)
            self.qMb_init_idxs = tf.expand_dims(self.qMb_init_idxs, 1)
            self.qMb_init_idxs_g12 = tf.concat([self.qMb_init_idxs for _ in range(0, self.num_hops)], axis=1)
            '''
            #^not consistently better, sometimes worse

            self.Mu_bs_variational_init_g12 = tf.einsum('bght,btgc->bghc', self.qMb_init_idxs_g12, self.Z_for_init_g12) # [b,g,h,c/g].
            self.qMb_init = self.get_variational_memory_init_provided_loc(batch_size, tf.stop_gradient(self.Mu_bs_variational_init_g12))

            # initialize local latents q(w_t)q(s_t).
            self.Mu_W_init = tf.zeros(dtype=tf.float32, shape=[batch_size, self.episode_len, self.num_hops, self.num_clusters, self.memory_size]) # [b, t, g, h, k]
            self.Sigma_w_init = tf.eye(self.memory_size, batch_shape=[batch_size, self.num_hops, self.num_clusters]) # [b, g, h, k, k]
            self.qs_probs_init = (1.0 / tf.cast(float(self.num_clusters), dtype=tf.float32)) * tf.ones(dtype=tf.float32, shape=[batch_size, self.episode_len, self.num_hops, self.num_clusters]) #[b,t,g,h]
            write_loop_init_vars = (0, self.qMb_init, self.qs_probs_init, self.Mu_W_init, self.Sigma_w_init)
            write_loop_cond = lambda i, *_: i < self.opt_iters
            def write_loop_body(i, qMb, qs_probs, Mu_W, Sigma_w):
                
                R = qMb.R
                U = qMb.U
                Mu_b = qMb.Mu_b
                Sigma_b = qMb.Sigma_b
                tau = (1.0 / tf.stop_gradient(self.sigma2))

                qs_probs_new = self.vb_update_qs(
                    Mu_Z=self.Mu_Z, R=R, U=U, Mu_b=Mu_b, Sigma_b=Sigma_b, Mu_W=Mu_W, Sigma_w=Sigma_w, tau=tau)

                Mu_b_new, Sigma_b_new = self.vb_update_qb(
                    Mu_Z=self.Mu_Z, R=R, U=U, qs=qs_probs_new, Mu_W=Mu_W, Sigma_w=Sigma_w, tau=tau)

                Mu_W_new, Sigma_w_new = self.vb_update_qwgivens(
                    Mu_Z=self.Mu_Z, R=R, U=U, Mu_b=Mu_b_new, Sigma_b=Sigma_b_new, tau=tau)

                R_new, U_new = self.vb_update_qM(
                    Mu_Z=self.Mu_Z, Mu_b=Mu_b_new, Sigma_b=Sigma_b_new, qs=qs_probs_new, Mu_W=Mu_W_new, Sigma_w=Sigma_w_new, tau=tau)

                return (i+1, MemoryState(R=R_new, U=U_new, Mu_b=Mu_b_new, Sigma_b=Sigma_b_new), qs_probs_new, Mu_W_new, Sigma_w_new)

            _, self.qMb, self.qs, self.Mu_W, self.Sigma_w = tf.while_loop(
                write_loop_cond, write_loop_body, write_loop_init_vars)

            # this nonsense is only used for convenience during deployment, you can ignore it for now.
            '''
            # (this version only works for num_hops G = 2, and doesnt always save space.)
            s1_sharp = tf.one_hot(tf.argmax(self.qs[:,:,0,:], axis=-1), depth=self.num_clusters)
            s2_sharp = tf.one_hot(tf.argmax(self.qs[:,:,1,:], axis=-1), depth=self.num_clusters)
            self.s1_probs = tf.reduce_mean(s1_sharp, axis=1) # [b,h].
            s12_sharp_counts = tf.einsum('bth,btj->bhj', s1_sharp, s2_sharp) # [b,h,h].
            self.s2_probs_given_s1 = s12_sharp_counts / tf.reduce_sum(s12_sharp_counts, axis=-1, keep_dims=True) # [b,h,h]. 
            '''
            self.sg_sharps = tf.one_hot(tf.argmax(self.qs, axis=-1), depth=self.num_clusters) # [B,T,G,H], contains a one-hot vector for each episode, timestep, and hop (indices B,T,G).

            ## prepare cholesky decompositions to sample M, b, w. very cheap to compute, since k is small. 
            self.chol_U = tf.linalg.cholesky(self.qMb.U) # [b, g, h, k, k]
            self.chol_Sigma_b = tf.linalg.cholesky(self.qMb.Sigma_b) # [b, g, h, c/g, c/g]
            self.chol_Sigma_w = tf.linalg.cholesky(self.Sigma_w) # [b, g, h, k, k]

            ## compute timestep-redundant terms of kl divergence for w_t
            #  will add the unique per-timestep term on later, and multiply the sum by 0.5.
            dkl_w_term_1 = tf.linalg.trace(self.Sigma_w)
            dkl_w_term_3 = -tf.cast(self.memory_size, dtype=tf.float32)
            dkl_w_term_4 = -2.0 * tf.reduce_sum(tf.log(tf.linalg.diag_part(self.chol_Sigma_w)), axis=-1)
            dkl_w_terms_134 = dkl_w_term_1 + dkl_w_term_3 + dkl_w_term_4

            ## read from memory
            z_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)
            z_kl_divs_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)
            w_kl_divs_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)
            s_kl_divs_array = tf.TensorArray(dtype=tf.float32, size=self.episode_len, infer_shape=True)

            read_loop_init_vars = (0, z_array, z_kl_divs_array, w_kl_divs_array, s_kl_divs_array)
            read_loop_cond = lambda t, *_: t < self.episode_len
            def read_loop_body(t, z_array, z_kl_divs_array, w_kl_divs_array, s_kl_divs_array):
                # q(s_t) parameters
                qst_probs = self.qs[:, t, :] # [b,g,h]

                # q(w_t|s_t) parameters for s_t = 1, ..., H.
                Mu_wt = self.Mu_W[:, t, :, :] # [b,g,h,k]

                # q(z_t) parameters
                Mu_zt_perceptual = self.Mu_Z[:, t, :]
                Mu_zt_perceptual_g12 = tf.reshape(Mu_zt_perceptual, [-1, self.num_hops, self.code_size // self.num_hops])
                Mu_zt_mem_g12 = tf.einsum('bgh,bghc->bgc', qst_probs, tf.einsum('bghkc,bghk->bghc', self.qMb.R, Mu_wt) + self.qMb.Mu_b) # [b,g,c/g].

                Mu_zt_g12 = self.apply_sr(Mu_zt_perceptual_g12, Mu_zt_mem_g12)
                Mu_zt = tf.reshape(Mu_zt_g12, [-1, self.code_size])
                Scale_diag_zt = self.Scale_diag_Z[:, t, :]

                q_zt = tfp.distributions.MultivariateNormalDiag(loc=Mu_zt, scale_diag=Scale_diag_zt)

                # sample zt for computing reconstruction likelihood.
                zt = q_zt.sample()

                # q(M), q(b), q(w_t|s_t) samples for computing z kl div.
                epsilon_M = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.num_hops, self.num_clusters, self.memory_size, self.code_size // self.num_hops]) # [b,g,h,k,c/g].
                M = self.qMb.R + tf.einsum('bghkl,bghlc->bghkc', self.chol_U, epsilon_M) # [b,g,h,k,c/g].
                epsilon_b = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.num_hops, self.num_clusters, self.code_size // self.num_hops]) # [b,g,h,c/g].
                b = self.qMb.Mu_b + tf.einsum('bghcz,bghz->bghc', self.chol_Sigma_b, epsilon_b) # [b,g,h,c/g].
                epsilon_wt = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.num_hops, self.num_clusters, self.memory_size]) # [b,g,h,k].
                wt = Mu_wt + tf.einsum('bghkl,bghl->bghk', self.chol_Sigma_w, epsilon_wt) # [b,g,h,k].

                # compute estimator of E_{q(M)q(b)q(s_t)q(w_t|s_t)} D_{KL}(q(z_t)||p(z_t|s_t,w_t,b,M)).
                q_Mu_zt = tf.reshape(Mu_zt, [-1, self.num_hops, self.code_size // self.num_hops]) # [b, g, c/g].
                q_Mu_zt_tiled = tf.tile(tf.expand_dims(q_Mu_zt, 2), multiples=[1, 1, self.num_clusters, 1]) # [b, g, h, c/g].

                q_Scale_diag_zt = tf.reshape(Scale_diag_zt, [-1, self.num_hops, self.code_size // self.num_hops]) # [b, g, c/g].
                q_Scale_diag_zt_tiled = tf.tile(tf.expand_dims(q_Scale_diag_zt, 2), multiples=[1, 1, self.num_clusters, 1]) # [b, g, h, c/g].

                p_Mu_zt = tf.einsum('bghkc,bghk->bghc', M, wt) + b # [b,g,h,c/g].
                p_Scale_diag_zt = tf.exp(self.pz_logsigma) * tf.ones_like(p_Mu_zt) # [b,g,h,c/g].

                dkl_zt_term1 = tf.einsum('bghc,bghc->bgh', (1.0 / tf.square(p_Scale_diag_zt)), tf.square(q_Scale_diag_zt_tiled)) # bgh
                dkl_zt_term2 = tf.einsum('bghc,bghc->bgh', tf.einsum('bghc,bghc->bghc', (q_Mu_zt_tiled - p_Mu_zt), (1.0 / tf.square(p_Scale_diag_zt))), (q_Mu_zt_tiled - p_Mu_zt)) # bgh
                dkl_zt_term3 = -tf.cast(self.code_size // self.num_hops, dtype=tf.float32) # scalar
                dkl_zt_term4a = -2.0 * tf.reduce_sum(tf.log(q_Scale_diag_zt_tiled), axis=-1) # bgh
                dkl_zt_term4b = 2.0 * tf.reduce_sum(tf.log(p_Scale_diag_zt), axis=-1) # bgh
                dkl_zt_by_hop_and_cluster = 0.5 * (dkl_zt_term1 + dkl_zt_term2 + dkl_zt_term3 + dkl_zt_term4a + dkl_zt_term4b) # bgh

                dkl_zt_by_hop = tf.einsum('bgh,bgh->bg', qst_probs, dkl_zt_by_hop_and_cluster) # bg.
                dkl_zt = tf.reduce_sum(dkl_zt_by_hop, axis=1) # b.
                # derived in App. E.

                # compute sum_g=1^G E_{q(s_t^(g))} D_{KL}(q(w_t^(g)|s_t^(g))||p(w_t^(g)|s_t^(g))).
                dkl_wt_term_2 = tf.reduce_sum(tf.square(Mu_wt), axis=-1) # bgh.
                dkl_wt_by_hop_and_cluster = 0.5 * (dkl_w_terms_134 + dkl_wt_term_2) # bgh.
                dkl_wt_by_hop = tf.einsum('bgh,bgh->bg', qst_probs, dkl_wt_by_hop_and_cluster) # bg.
                dkl_wt = tf.reduce_sum(dkl_wt_by_hop, axis=1) # b.

                # s_kl_div computation.
                pst_probs = (1.0 / tf.cast(float(self.num_clusters), dtype=tf.float32)) * tf.ones(dtype=tf.float32, shape=[batch_size, self.num_hops, self.num_clusters])
                dkl_st_by_hop = tf.reduce_sum(qst_probs * tf.log(qst_probs), axis=1) - tf.log(pst_probs[:, 0]) # uniform prior
                dkl_st = tf.reduce_sum(dkl_st_by_hop, axis=-1)

                return (t+1, z_array.write(t, zt), z_kl_divs_array.write(t, dkl_zt), w_kl_divs_array.write(t, dkl_wt), s_kl_divs_array.write(t, dkl_st))

            _, z_samples, z_kl_divs, w_kl_divs, s_kl_divs = tf.while_loop(read_loop_cond, read_loop_body, read_loop_init_vars)

            self.z_samples = tf.transpose(z_samples.stack(), [1, 0, 2])  # [B, T, C]
            self.z_kl_divs = tf.transpose(z_kl_divs.stack())  # [B, T]
            self.w_kl_divs = tf.transpose(w_kl_divs.stack())  # [B, T]
            self.s_kl_divs = tf.transpose(s_kl_divs.stack())  # [B, T]

            # decode
            self.z_samples_batched = tf.reshape(self.z_samples, [-1, self.code_size])  # [B*T, C]
            self.px_given_z_batched = self.px_given_z(self.z_samples_batched, training=self.is_train)  # batch_shape [B*T], element_shape [H, W, C]
            self.px_given_z_ = tfp.distributions.BatchReshape(self.px_given_z_batched, batch_shape=[-1, self.episode_len])  # batch_shape [B, T], element_shape [H, W, C]

            self.log_probs_x_given_z = self.px_given_z_.log_prob(self.input_xs)

            # dkl_M
            term1 = tf.cast(self.code_size // self.num_hops, dtype=tf.float32) * tf.linalg.trace(tf.einsum('bghkl,bghlp->bghkp', self.U0_invs, self.qMb.U)) # [b,g,h].
            term2 = tf.linalg.trace(tf.einsum('bghkc,bghkz->bghcz', (self.qMb.R - self.pMb.R), tf.einsum('bghkl,bghlc->bghkc', self.U0_invs, (self.qMb.R - self.pMb.R)))) # [b,g,h].
            term3 = -tf.cast((self.memory_size * self.code_size // self.num_hops), dtype=tf.float32) # scalar
            term4 = -2.0 * tf.cast(self.code_size // self.num_hops, dtype=tf.float32) * tf.reduce_sum(tf.log(tf.linalg.diag_part(self.chol_U)), axis=-1) # shape [b, g, h] from [b, g, h, k, k].
            term5 = 1.0 * tf.cast(self.code_size // self.num_hops, dtype=tf.float32) * tf.reduce_sum(tf.log(tf.linalg.diag_part(self.pMb.U)), axis=-1) # shape [b, g, h] from [b, g, h, k, k]
            self.dkl_M_by_hop_and_cluster = 0.5 * (term1 + term2 + term3 + term4 + term5) # shape [b, g, h]
            self.dkl_M_by_hop = tf.reduce_sum(self.dkl_M_by_hop_and_cluster, axis=-1) # shape [b, g].
            self.dkl_M_total = tf.reduce_sum(self.dkl_M_by_hop, axis=-1) # shape [b]

            # dkl_b
            term1 = tf.linalg.trace(tf.einsum('bghcz,bghzq->bghcq', self.Sigma_b0_invs, self.qMb.Sigma_b)) # [b,g,h].
            term2 = tf.einsum('bghc,bghc->bgh', (self.qMb.Mu_b - self.pMb.Mu_b), tf.einsum('bghcz,bghz->bghc', self.Sigma_b0_invs, (self.qMb.Mu_b - self.pMb.Mu_b))) # [b,g,h].
            term3 = -tf.cast(self.code_size // self.num_hops, dtype=tf.float32) # scalar
            term4 = -2.0 * tf.reduce_sum(tf.log(tf.linalg.diag_part(self.chol_Sigma_b)), axis=-1) # shape [b, g, h] from [b, g, h, c/g, c/g]
            term5 = 1.0 * tf.reduce_sum(tf.log(tf.linalg.diag_part(self.Sigma_b0s)), axis=-1) # shape [b, g, h] from [b, g, h, c/g, c/g]
            self.dkl_b_by_hop_and_cluster = 0.5 * (term1 + term2 + term3 + term4 + term5) # shape [b, g, h]
            self.dkl_b_by_hop = tf.reduce_sum(self.dkl_b_by_hop_and_cluster, axis=-1) # shape [b, g].
            self.dkl_b_total = tf.reduce_sum(self.dkl_b_by_hop, axis=-1) # shape [b]

            # compute the elbo objective
            self.elbo_episode = tf.reduce_sum(
                (self.log_probs_x_given_z - self.z_kl_divs - self.w_kl_divs - self.s_kl_divs), axis=1) - self.dkl_M_total - self.dkl_b_total # [B]

            self.elbo = tf.reduce_mean(self.elbo_episode, axis=0)  # []
            self.elbo_per_frame = self.elbo / tf.cast(self.episode_len, dtype=tf.float32)

            self.loss = -self.elbo_per_frame

            # optimizer
            self.optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.0)
            tvars = [v for v in tf.trainable_variables() if v.name.startswith(self._name)]
            self.gradients, _ = zip(*self.optimizer.compute_gradients(self.loss, tvars))
            self.gradients = [tf.where(tf.logical_or(tf.math.is_inf(g), tf.math.is_nan(g)), tf.zeros_like(v), g) for g,v in zip(self.gradients, tvars)]
            self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, 100.0)

            # use control dependencies on update ops - this is required by batchnorm.
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = self.optimizer.apply_gradients(
                    grads_and_vars=zip(self.clipped_gradients, tvars),
                    global_step=self.global_step)

            ## tensorboard summaries
            self.dkl_z_per_frame = tf.reduce_mean(self.z_kl_divs, axis=[0, 1])
            self.dkl_w_per_frame = tf.reduce_mean(self.w_kl_divs, axis=[0, 1])
            self.dkl_s_per_frame = tf.reduce_mean(self.s_kl_divs, axis=[0, 1])
            self.dkl_M_total_per_frame = tf.reduce_mean(self.dkl_M_total, axis=0) / tf.cast(self.episode_len, dtype=tf.float32)
            self.dkl_b_total_per_frame = tf.reduce_mean(self.dkl_b_total, axis=0) / tf.cast(self.episode_len, dtype=tf.float32)

            tf.summary.scalar('elbo_per_frame', self.elbo_per_frame)
            tf.summary.scalar('dkl_z_per_frame', self.dkl_z_per_frame)
            tf.summary.scalar('dkl_w_per_frame', self.dkl_w_per_frame)
            tf.summary.scalar('dkl_s_per_frame', self.dkl_s_per_frame)
            tf.summary.scalar('dkl_M_total_per_frame', self.dkl_M_total_per_frame)
            tf.summary.scalar('dkl_b_total_per_frame', self.dkl_b_total_per_frame)
            tf.summary.scalar('pz_sigma_squared', tf.exp(2.0 * self.pz_logsigma))
            tf.summary.scalar('pb_sigma_squared', tf.exp(2.0 * self.pb_logsigma))

            for g in range(0, self.num_hops):
                for h in range(0, self.num_clusters):
                    tf.summary.scalar('hop_{}_cluster_{}_usage'.format(g, h), tf.reduce_mean(self.qs[:,:,g,h], axis=[0,1]))
                    # note that this is average cluster assignment value per batch/timestep item, but doesnt say if the clusters are actually diverse.

            self.merged_summaries = tf.summary.merge_all()

            ## misc ops - not used during training.
            #  using a given memory state:
            self.input_R = tf.placeholder(tf.float32, shape=[None, self.num_hops, self.num_clusters, self.memory_size, self.code_size // self.num_hops])
            self.input_U = tf.placeholder(tf.float32, shape=[None, self.num_hops, self.num_clusters, self.memory_size, self.memory_size])
            self.input_Mu_b = tf.placeholder(tf.float32, shape=[None, self.num_hops, self.num_clusters, self.code_size // self.num_hops])
            self.input_Sigma_b = tf.placeholder(tf.float32, shape=[None, self.num_hops, self.num_clusters, self.code_size // self.num_hops, self.code_size // self.num_hops])
            self.memory_state = MemoryState(R=self.input_R, U=self.input_U, Mu_b=self.input_Mu_b, Sigma_b=self.input_Sigma_b)

            '''
            self.input_s1_probs = tf.placeholder(tf.float32, shape=[None, self.num_clusters])
            self.input_s2_probs_given_s1 = tf.placeholder(tf.float32, shape=[None, self.num_clusters, self.num_clusters])
            '''
            self.input_sg_sharps = tf.placeholder(tf.float32, shape=[None, self.episode_len, self.num_hops, self.num_clusters]) # [B,T,G,H]. 
            # hard assignments tensor corresponding to the observations in an episode. the empirical pseudocounts to be used when generating directly, they need to be stored in some way.
            # for simplicity of implementation we will just sample uniformly from axis T instead of creating a datastructure to store the pseudocounts for hard assignments.
            # the result will be the same; only issue is space complexity here can be higher (sometimes). 

            # generate - sample p(w), p(z|w,M), p(x|z)
            #def generate(qMb, s1_probs, s2_probs_given_s1):
            def generate(qMb, sg_sharps):
                batch_size = tf.shape(qMb.R)[0]
                R = qMb.R
                U = qMb.U
                Mu_b = qMb.Mu_b
                Sigma_b = qMb.Sigma_b

                '''
                s1_dist = tfp.distributions.OneHotCategorical(probs=s1_probs, dtype=tf.float32)
                s1 = s1_dist.sample() # [b,h]
                s2_dist = tfp.distributions.OneHotCategorical(probs=tf.einsum('bh,bhj->bj', s1, s2_probs_given_s1), dtype=tf.float32)
                s2 = s2_dist.sample() # [b,h]
                s = tf.concat([tf.expand_dims(s1, 1), tf.expand_dims(s2, 1)], axis=1) # [b,g,h].
                print(s.get_shape().as_list())
                '''
                t_dist = tfp.distributions.OneHotCategorical(probs=(1.0/float(self.episode_len))*tf.ones(dtype=tf.float32,shape=[batch_size,self.episode_len]), dtype=tf.float32)
                t = t_dist.sample() # [b,t].
                s = tf.einsum('btgh,bt->bgh', sg_sharps, t) # [b,g,h].

                R = tf.einsum('bgh,bghkc->bgkc', s, R) # [b,g,k,c/g].
                U = tf.einsum('bgh,bghkl->bgkl', s, U) # [b,g,k,k].
                Mu_b = tf.einsum('bgh,bghc->bgc', s, Mu_b) # [b,g,c/g].
                Sigma_b = tf.einsum('bgh,bghcz->bgcz', s, Sigma_b) # [b,g,c/g,c/g]. 

                chol_U = tf.linalg.cholesky(U)
                epsilon_M = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.num_hops, self.memory_size, self.code_size // self.num_hops])
                M = R + tf.einsum('bgkl,bglc->bgkc', chol_U, epsilon_M) # [b,g,k,c/g].
                M = tf.transpose(M, perm=[0, 2, 1, 3]) # [b,k,g,c/g].
                M = tf.reshape(M, [-1, self.memory_size, self.code_size])

                chol_Sigma_b = tf.linalg.cholesky(Sigma_b)
                epsilon_b = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.num_hops, self.code_size // self.num_hops])
                b = Mu_b + tf.einsum('bgcz,bgz->bgc', chol_Sigma_b, epsilon_b) # [b,g,c/g].
                b = tf.reshape(b, [-1, self.code_size])

                pw = self.pw(batch_size=batch_size)
                w = pw.sample()
                pz = self.pz_given_wMb(w, M, b, self.pz_logsigma)
                z = pz.sample()
                px = self.px_given_z(z, training=False)
                x = px.sample()

                return x

            #self.generated_x = generate(self.memory_state, self.input_s1_probs, self.input_s2_probs_given_s1)
            self.generated_x = generate(self.memory_state, self.input_sg_sharps)

            # read - given q(z) perceptual, infer q(s), q(w|s); then sample q(s), q(w|s), q(M), q(b), p(z|w, M_s, b_s), p(x|z)
            def read(query_x, qMb):
                batch_size = tf.shape(qMb.R)[0]
                R = qMb.R
                U = qMb.U
                Mu_b = qMb.Mu_b
                Sigma_b = qMb.Sigma_b

                qz_perceptual = self.qz(query_x, training=False)
                Mu_z_perceptual = qz_perceptual.mean()
                Mu_z_perceptual_g12 = tf.reshape(Mu_z_perceptual, [-1, self.num_hops, self.code_size // self.num_hops])

                # optimize variational distributions q(s)q(w) for local latent variables y = {s, w} for reading from memory Omega = {M_1:H, b_1:H}. 
                Mu_w_init = tf.zeros(dtype=tf.float32, shape=[batch_size, self.num_hops, self.num_clusters, self.memory_size])
                Sigma_w_init = tf.eye(self.memory_size, batch_shape=[batch_size, self.num_hops, self.num_clusters])
                qs_probs_init = (1.0 / float(self.num_clusters)) * tf.ones(dtype=tf.float32, shape=[batch_size, self.num_hops, self.num_clusters])
                CdivG = tf.cast(self.code_size // self.num_hops, dtype=tf.float32)
                tau = (1.0 / tf.stop_gradient(self.sigma2))

                def addressing_loop_body(i, qs_probs, Mu_w, Sigma_w):
                    RRT_plus_CdivGU = tf.einsum('bghkc,bghlc->bghkl', R, R) + CdivG * U

                    qs_logits_term_1a = tau * tf.einsum('bghk,bghk->bgh', Mu_w, tf.einsum('bghkl,bghl->bghk', RRT_plus_CdivGU, Mu_w))
                    qs_logits_term_1b = tau * tf.linalg.trace(tf.einsum('bghkl,bghlp->bghkp', RRT_plus_CdivGU, Sigma_w))
                    qs_logits_term_1c = tau * tf.einsum('bghc,bghc->bgh', Mu_b, Mu_b)
                    qs_logits_term_1d = tau * tf.linalg.trace(Sigma_b)
                    qs_logits_term_1 = -0.5 * (qs_logits_term_1a + qs_logits_term_1b + qs_logits_term_1c + qs_logits_term_1d)
    
                    Mu_z_minus_Mu_bh = tf.expand_dims(Mu_z_perceptual_g12, 2) - Mu_b # [b,g,h,c/g].
                    qs_logits_term_2 = tau * tf.einsum('bghk,bghk->bgh', Mu_w, tf.einsum('bghkc,bghc->bghk', R, Mu_z_minus_Mu_bh)) # [b, g, h].
                    qs_logits_term_3 = tau * tf.einsum('bghc,bgc->bgh', Mu_b, Mu_z_perceptual_g12) # [b, g, h].
                    qs_logits_term_4 = 0.5 * (2 * tf.reduce_sum(tf.log(tf.linalg.diag_part(tf.linalg.cholesky(Sigma_w))), axis=-1)) # [b, g, h].
                    qs_logits_term_5 = -0.5 * (tf.einsum('bghk,bghk->bgh', Mu_w, Mu_w) + tf.linalg.trace(Sigma_w)) # [b, g, h].
                    qs_logits = qs_logits_term_1 + qs_logits_term_2 + qs_logits_term_3 + qs_logits_term_4 + qs_logits_term_5 # [b, g, h].
                    qs_probs_new = tf.nn.softmax(qs_logits, axis=-1) # [b,g,h].
                    qs_probs_new = (1e-3 + qs_probs_new) / tf.reduce_sum((1e-3 + qs_probs_new), axis=-1, keep_dims=True) # [b,g,h].

                    Lambda_w_new = tf.eye(self.memory_size, batch_shape=[batch_size, self.num_hops, self.num_clusters]) + tau * RRT_plus_CdivGU # [b,g,h,k].
                    Sigma_w_new = tf.linalg.inv(Lambda_w_new) # [b,g,h,k,k].
                    Eta_w_new = tau * tf.einsum('bghkc,bghc->bghk', R, (tf.expand_dims(Mu_z_perceptual_g12, 2) - Mu_b)) # [b,g,h,k].
                    Mu_w_new = tf.einsum('bghkl,bghl->bghk', Sigma_w_new, Eta_w_new) # [b,g,h,k].

                    return (i+1, qs_probs_new, Mu_w_new, Sigma_w_new)

                addressing_loop_init = (0, qs_probs_init, Mu_w_init, Sigma_w_init)
                addressing_loop_cond = lambda i, *_: i < self.opt_iters
                _, qs_probs, Mu_w, Sigma_w = tf.while_loop(addressing_loop_cond, addressing_loop_body, addressing_loop_init)

                qs = tfp.distributions.OneHotCategorical(probs=qs_probs, dtype=tf.float32)
                s = qs.sample()

                chol_Sigma_w = tf.linalg.cholesky(Sigma_w)
                epsilon_w = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.num_hops, self.num_clusters, self.memory_size])
                w = Mu_w + tf.einsum('bghkl,bghl->bghk', chol_Sigma_w, epsilon_w)

                chol_U = tf.linalg.cholesky(U)
                epsilon_M = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.num_hops, self.num_clusters, self.memory_size, self.code_size // self.num_hops])
                M = R + tf.einsum('bghkl,bghlc->bghkc', chol_U, epsilon_M)

                chol_Sigma_b = tf.linalg.cholesky(Sigma_b)
                epsilon_b = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.num_hops, self.num_clusters, self.code_size // self.num_hops])
                b = Mu_b + tf.einsum('bghcz,bghz->bghc', chol_Sigma_b, epsilon_b)

                w = tf.einsum('bgh,bghk->bgk', s, w)
                M = tf.einsum('bgh,bghkc->bgkc', s, M)
                b = tf.einsum('bgh,bghc->bgc', s, b)

                p_Mu_z_g12 = tf.einsum('bgkc,bgk->bgc', M, w) + b # [b,g,c/g].
                p_Scale_diag_z_g12 = tf.exp(self.pz_logsigma) * tf.ones_like(p_Mu_z_g12)
                epsilon_pz_g12 = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, self.num_hops, self.code_size // self.num_hops])
                pz_sampled_g12 = p_Mu_z_g12 + tf.einsum('bgc,bgc->bgc', p_Scale_diag_z_g12, epsilon_pz_g12) # # [b,g,c/g].

                '''
                qz_sampled = qz_perceptual.sample()
                qz_sampled_g12 = tf.reshape(qz_sampled, [-1, self.num_hops, self.code_size // self.num_hops])

                # modify a random code block using the memory. does it work?
                hop_probs = (1.0 / float(self.num_hops)) * tf.tile(tf.ones(dtype=tf.float32, shape=[1, self.num_hops]), multiples=[batch_size, 1])
                hop_dist = tfp.distributions.OneHotCategorical(probs=hop_probs, dtype=tf.float32)
                hop_onehot = hop_dist.sample()

                z_g12 = tf.einsum('bg,bgc->bgc', 1.0-hop_onehot, qz_sampled_g12) + tf.einsum('bg,bgc->bgc', hop_onehot, pz_sampled_g12) # [b,g,c/g].
                z = tf.reshape(z_g12, [-1, self.code_size])
                '''
                z_g12 = pz_sampled_g12
                z = tf.reshape(z_g12, [-1, self.code_size])

                px = self.px_given_z(z, training=False)
                x = px.mean() if self.discrete_outputs else px.sample()
                return x, z

            self.query_x = tf.placeholder(tf.float32, shape=[None, self.img_height, self.img_width, self.img_channels])
            self.read_x, self.read_z = read(self.query_x, self.memory_state)

            # copy - sample q(z), p(x|z)
            def copy(query_x):
                batch_size = tf.shape(query_x)[0]
                qz = self.qz(query_x, training=False)
                z = qz.sample()
                px = self.px_given_z(z, training=False)
                x = px.mean() if self.discrete_outputs else px.sample()
                return x, z

            self.copied_x, self.copied_z = copy(self.query_x)

    def get_memory_prior(self, batch_size):
        with tf.variable_scope('initial_state', reuse=tf.AUTO_REUSE):
            # initialize prior params for global latent variables. 
            # this code supports dynamically resizable memory, in the sense of restoring a checkpoint with a larger self.num_hops, self.num_clusters, or self.memory_size.
            # modifying self.num_clusters is the most sensible option.

            R_0 = tf.zeros(dtype=tf.float32, shape=[self.num_hops, self.memory_size, self.code_size // self.num_hops]) # [g,k,c/g].
            U_0 = (self.pM_sigma2 / tf.cast(float(self.memory_size), dtype=tf.float32)) * tf.eye(self.memory_size, batch_shape=[self.num_hops]) # [g,k,k].
            Mu_b0 = tf.zeros(dtype=tf.float32, shape=[self.num_hops, self.code_size // self.num_hops]) # [g,c/g].
            Sigma_b0 = self.pb_sigma2 * tf.eye(self.code_size // self.num_hops, batch_shape=[self.num_hops]) # [g,c/g,c/g].

            R_0 = tf.reshape(R_0, [1, self.num_hops, 1, self.memory_size, self.code_size // self.num_hops]) # [1, g, 1, k, c/g].
            U_0 = tf.reshape(U_0, [1, self.num_hops, 1, self.memory_size, self.memory_size]) # [1, g, 1, k, k].
            Mu_b0 = tf.reshape(Mu_b0, [1, self.num_hops, 1, self.code_size // self.num_hops]) # [1, g, 1, c/g].
            Sigma_b0 = tf.reshape(Sigma_b0, [1, self.num_hops, 1, self.code_size // self.num_hops, self.code_size // self.num_hops]) # [1, g, 1, c/g, c/g].

            R_0 = tf.tile(R_0, [batch_size, 1, self.num_clusters, 1, 1]) # shape [b,g,h,k,c/g]
            U_0 = tf.tile(U_0, [batch_size, 1, self.num_clusters, 1, 1]) # shape [b,g,h,k,k]
            Mu_b0 = tf.tile(Mu_b0, [batch_size, 1, self.num_clusters, 1]) # shape [b,g,h,c/g]
            Sigma_b0 = tf.tile(Sigma_b0, [batch_size, 1, self.num_clusters, 1, 1]) # shape [b,g,h,c/g,c/g]

            return MemoryState(R=R_0, U=U_0, Mu_b=Mu_b0, Sigma_b=Sigma_b0)

    def get_variational_memory_init_provided_loc(self, batch_size, loc_means_g12):
        # randomly initialized q(M1:H)q(b1:H). VBEM generally uses a randomly initialized variational posterior instead of randomizing the prior.
        # this allows symmetry breaking during inference without requiring an asymmetric prior.
        # this initialization is a highly heuristic one.
        #
        # note that we also pass in the k-means++ init here for the means of each q(b_h^(g)), the variational posterior over per-hop cluster locations.

        R = tf.random.normal(mean=0.0, stddev=(1.0 / float(self.code_size // self.num_hops)), 
            shape=[batch_size, self.num_hops, self.num_clusters, self.memory_size, self.code_size // self.num_hops]) # [b,g,h,k,c/g].
        U = (1.0 / float(self.code_size // self.num_hops)) * tf.eye(self.memory_size, batch_shape=[batch_size, self.num_hops, self.num_clusters]) # shape [b,g,h,k,k].

        Mu_b = tf.stop_gradient(loc_means_g12) # [b,g,h,c/g].
        Sigma_b = tf.stop_gradient(tf.exp(2.0 * self.pb_logsigma)) * tf.eye(self.code_size // self.num_hops, batch_shape=[batch_size, self.num_hops, self.num_clusters]) # shape [b,g,h,c/g,c/g].

        return MemoryState(R=R, U=U, Mu_b=Mu_b, Sigma_b=Sigma_b)

    def vb_update_qs(self, Mu_Z, R, U, Mu_b, Sigma_b, Mu_W, Sigma_w, tau):
        # computes vb update for q(s_t^(g)).
        CdivG = tf.cast(self.code_size // self.num_hops, dtype=tf.float32)
        RRT_plus_CdivGU = tf.einsum('bghkc,bghlc->bghkl', R, R) + CdivG * U # RhRh^T + (C/G)*Uh

        qs_logits_term_1a = tau * tf.einsum('btghk,btghk->btgh', Mu_W, tf.einsum('bghkl,btghl->btghk', RRT_plus_CdivGU, Mu_W)) # shape: [b,t,g,h]
        qs_logits_term_1b = tau * tf.linalg.trace(tf.einsum('bghkl,btghlp->btghkp', RRT_plus_CdivGU, tf.expand_dims(Sigma_w, 1))) # shape: [b,1,g,h]
        qs_logits_term_1c = tau * tf.expand_dims(tf.einsum('bghc,bghc->bgh', Mu_b, Mu_b), 1) # shape: [b,1,g,h]. 
        qs_logits_term_1d = tau * tf.expand_dims(tf.linalg.trace(Sigma_b), 1) # shape: [b,1,g,h]. 
        qs_logits_term_1 = -0.5 * (qs_logits_term_1a + qs_logits_term_1b + qs_logits_term_1c + qs_logits_term_1d) # shape [b,t,g,h]

        Mu_Z_g12 = tf.reshape(Mu_Z, [-1, self.episode_len, self.num_hops, self.code_size // self.num_hops])
        Mu_Z_minus_Mu_bh = tf.expand_dims(Mu_Z_g12, 3) - tf.expand_dims(Mu_b, 1) # shape: [b,t,g,h,c/g]. 
        qs_logits_term_2 = tau * tf.einsum('btghk,btghk->btgh', Mu_W, tf.einsum('bghkc,btghc->btghk', R, Mu_Z_minus_Mu_bh)) # shape: [b,t,g,h]. 
        qs_logits_term_3 = tau * tf.einsum('bghc,btgc->btgh', Mu_b, Mu_Z_g12) # shape: [b,t,g,h]. 
        qs_logits_term_4 = tf.expand_dims(2 * tf.reduce_sum(tf.log(1e-6 + tf.linalg.diag_part(tf.linalg.cholesky(Sigma_w))), axis=-1), 1) # shape: [b,1,g,h]. 
        qs_logits_term_5 = -0.5 * (tf.einsum('btghk,btghk->btgh', Mu_W, Mu_W) + tf.expand_dims(tf.linalg.trace(Sigma_w), 1)) # shape: [b,t,g,h]. 
        
        qs_logits = qs_logits_term_1 + qs_logits_term_2 + qs_logits_term_3 + qs_logits_term_4 + qs_logits_term_5
        qs_probs_new = tf.nn.softmax(qs_logits, axis=-1) # shape [b,t,g,h]
        qs_probs_new = (1e-3 + qs_probs_new) / tf.reduce_sum((1e-3 + qs_probs_new), axis=-1, keep_dims=True)

        return qs_probs_new

    def vb_update_qb(self, Mu_Z, R, U, qs, Mu_W, Sigma_w, tau):
        # computes vb update for q(b_h^(g)).
        batch_size = tf.shape(Mu_Z)[0]

        I_CdivG = tf.eye(self.code_size // self.num_hops, batch_shape=[batch_size, self.num_hops, self.num_clusters])
        Lambda_b_scalars = (1.0 / self.pb_sigma2) + tau * tf.reduce_sum(qs, axis=1) # shape: [b,g,h].
        Sigma_b_scalars = (1.0 / Lambda_b_scalars) # shape: [b,g,h].
        Sigma_b_new = tf.einsum('bgh,bghcz->bghcz', Sigma_b_scalars, I_CdivG) # shape: [b,g,h,c/g,c/g].

        Mu_Z_g12 = tf.reshape(Mu_Z, [-1, self.episode_len, self.num_hops, self.code_size // self.num_hops]) # [b,t,g,c/g].
        Muzt_minus_RT_Muwt = tf.expand_dims(Mu_Z_g12, 3) - tf.einsum('bghkc,btghk->btghc', R, Mu_W) # shape: [b,t,g,h,c/g].
        weighted_summed_Muzt_minus_RT_Muwt = tf.einsum('btgh,btghc->bghc', qs, Muzt_minus_RT_Muwt) # shape: [b,g,h,c/g].
        Eta_b_new = tf.stop_gradient(self.Sigma_b0_inv_Mu_b0s) + tau * weighted_summed_Muzt_minus_RT_Muwt # shape: [b,g,h,c/g].
        Mu_b_new = tf.einsum('bghcz,bghz->bghc', Sigma_b_new, Eta_b_new) # shape: [b,g,h,c/g].

        return Mu_b_new, Sigma_b_new

    def vb_update_qwgivens(self, Mu_Z, R, U, Mu_b, Sigma_b, tau):
        # computes vb update for q(w_t^(g)|s_t^(g)).
        batch_size = tf.shape(Mu_Z)[0]
        CdivG = tf.cast(self.code_size // self.num_hops, dtype=tf.float32)

        I_K = tf.eye(self.memory_size, batch_shape=[batch_size, self.num_hops, self.num_clusters])
        RRT_plus_CdivGU = tf.einsum('bghkc,bghlc->bghkl', R, R) + CdivG * U # shape [b,g,h,k,k]. 
        Lambda_w_new = I_K + tau * RRT_plus_CdivGU # shape [b,g,h,k,k].
        Sigma_w_new = tf.linalg.inv(Lambda_w_new) # shape [b,g,h,k,k].

        Mu_Z_g12 = tf.reshape(Mu_Z, [-1, self.episode_len, self.num_hops, self.code_size // self.num_hops]) # [b,t,g,c/g].
        Muzt_minus_Mubh = tf.expand_dims(Mu_Z_g12, 3) - tf.expand_dims(Mu_b, 1) # shape: [b,t,g,h,c/g].
        Eta_W_new = tau * tf.einsum('bghkc,btghc->btghk', R, Muzt_minus_Mubh) # shape: [b,t,g,h,k].
        Mu_W_new = tf.einsum('bghkl,btghl->btghk', Sigma_w_new, Eta_W_new) # shape: [b,t,g,h,k].

        return Mu_W_new, Sigma_w_new

    def vb_update_qM(self, Mu_Z, Mu_b, Sigma_b, qs, Mu_W, Sigma_w, tau):
        # computes vb update for q(M_h^(g)).
        weighted_summed_MuwMuwT = tf.einsum('btghk,btghl->bghkl', tf.einsum('btgh,btghk->btghk', qs, Mu_W), Mu_W) # more efficient 
        weighted_summed_Sigmaw = tf.einsum('bgh,bghkl->bghkl', tf.reduce_sum(qs, axis=1), Sigma_w) # more efficient
        weighted_summed_MuwMuwTplusSigmaw = weighted_summed_MuwMuwT + weighted_summed_Sigmaw
        Lambda_M_new = tf.stop_gradient(self.U0_invs) + tau * weighted_summed_MuwMuwTplusSigmaw
        U_new = tf.linalg.inv(Lambda_M_new)

        Mu_Z_g12 = tf.reshape(Mu_Z, [-1, self.episode_len, self.num_hops, self.code_size // self.num_hops]) # [b,t,g,c/g].
        weighted_Muw = tf.einsum('btgh,btghk->btghk', qs, Mu_W) # [b,t,g,h,k].
        weighted_summed_MuwMuzT = tf.einsum('btghk,btgc->bghkc', weighted_Muw, Mu_Z_g12) # [b,g,h,k,c/g].
        weighted_summed_MuwMubhT = tf.einsum('btghk,bghc->bghkc', weighted_Muw, Mu_b) # [b,g,h,k,c/g].
        weighted_summed_Muw_outer_MuzminusMubT = weighted_summed_MuwMuzT - weighted_summed_MuwMubhT # [b,g,h,k,c/g].

        Eta_M_new = tf.stop_gradient(self.U0_inv_R0s) + tau * weighted_summed_Muw_outer_MuzminusMubT  # [b,g,h,k,c/g].
        R_new = tf.einsum('bghkl,bghlc->bghkc', U_new, Eta_M_new)  # [b,g,h,k,c/g].

        return R_new, U_new

    def qz(self, input_x, training=True):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            input_x = input_x if self.discrete_outputs else (2.0 * input_x - 1.0)
            blocks = [input_x]

            for i in range(0, self.enc_blocks):
                block = self.encoder_block(blocks[i], training=training, name='enc_block_' + str(i + 1))
                blocks.append(block)

            last = blocks[self.enc_blocks]
            last = tf.layers.flatten(last)
            encoded_x = tf.layers.dense(
                last, units=(2 * self.code_size), use_bias=False, activation=None) 

            fc1 = tf.layers.dense(encoded_x, units=(2 * self.code_size), use_bias=True, activation=self.activation)
            fc2 = tf.layers.dense(fc1, units=self.code_size, use_bias=False, activation=None)
            mu = fc2
            scale_diag = tf.sqrt(0.5 * self.pz_sigma2) * tf.ones_like(mu) # entirely heuristic. 
            z_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=scale_diag)
            z_dist = tfp.distributions.Independent(z_dist)
            return z_dist

    def px_given_z(self, z, training=True):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            s = (2 ** self.dec_blocks)
            dec_init_h = int(np.ceil(self.img_height / float(s)))
            dec_init_w = int(np.ceil(self.img_width / float(s)))
            fc_units = int(dec_init_h * dec_init_w * self.num_filters)

            fc = tf.layers.dense(z, units=fc_units, use_bias=False, activation=None, kernel_initializer=tf.orthogonal_initializer())
            fc_3d = tf.reshape(fc, shape=[-1, dec_init_h, dec_init_w, self.num_filters])

            blocks = [fc_3d]
            for i in range(0, self.dec_blocks):
                block = self.decoder_block(blocks[i], training=training, name='dec_block_' + str(i + 1))
                blocks.append(block)

            last = blocks[self.dec_blocks]
            last = self.maybe_center_crop(last, self.img_height, self.img_width)

            if self.discrete_outputs:
                # this is to support binarized image data only
                decoded_logits_x = tf.layers.conv2d(
                    last, filters=self.img_channels,
                    kernel_size=1, strides=1, padding='same', activation=None)

                x_dist = tfp.distributions.Bernoulli(logits=decoded_logits_x)
                x_dist = tfp.distributions.Independent(x_dist)
                return x_dist
            else:
                # this is stable, and worked better than using a non-residual block of stride > 1.  
                decoded_mu_x = tf.layers.conv2d(
                    last, filters=self.img_channels,
                    kernel_size=1, strides=1, padding='same', activation=tf.nn.sigmoid)

                decoded_sigma_x = 5e-3 + 0.50 * tf.layers.conv2d(
                    last, filters=self.img_channels,
                    kernel_size=1, strides=1, padding='same', activation=tf.nn.sigmoid)

                x_dist = tfp.distributions.MultivariateNormalDiag(loc=decoded_mu_x, scale_diag=decoded_sigma_x)
                x_dist = tfp.distributions.Independent(x_dist)
                return x_dist

    def maybe_center_crop(self, tensor, target_height, target_width):
        with tf.variable_scope('maybe_center_crop', reuse=tf.AUTO_REUSE):
            # assumes tensor shape is such that the extra space is divisible by two
            shape = tensor.get_shape().as_list()
            print(shape)
            h, w = shape[1], shape[2]
            h_slack = h - target_height
            w_slack = w - target_width
            h_start_idx = 0 + tf.cast(h_slack / 2, dtype=tf.int32)
            h_end_idx = h - tf.cast(h_slack / 2, dtype=tf.int32)
            w_start_idx = 0 + tf.cast(w_slack / 2, dtype=tf.int32)
            w_end_idx = w - tf.cast(w_slack / 2, dtype=tf.int32)
            tensor_maybe_cropped = tensor[:, h_start_idx:h_end_idx, w_start_idx:w_end_idx, :]
            return tensor_maybe_cropped

    def encoder_block(self, inputs, training, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            glorot_init = tf.glorot_normal_initializer()

            conv = tf.layers.conv2d(
                inputs, filters=self.num_filters, kernel_size=4, strides=2, padding='valid', 
                kernel_initializer=glorot_init, activation=None)

            res_blocks = [conv]
            for j in range(0, self.res_blocks_per_encoder_block):
                res_block = self.residual_block(res_blocks[j], training=training, name='res_block_' + str(j+1))
                res_blocks.append(res_block)

            last = res_blocks[self.res_blocks_per_encoder_block]
            output = last

            return output

    def group_norm(self, inputs, num_groups, name):
        channels = inputs.get_shape().as_list()[-1]
        assert channels % num_groups == 0

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            group_size = channels // num_groups
            loc = tf.get_variable(name='loc', shape=[num_groups, group_size], initializer=tf.constant_initializer(value=0.0))
            scale = tf.get_variable(name='scale', shape=[num_groups, group_size], initializer=tf.constant_initializer(value=1.0))

            good_shape = [-1] + inputs.get_shape().as_list()[1:3] + [num_groups, (channels // num_groups)]
            inputs_by_group = tf.reshape(inputs, good_shape) # [b,h,w,g,f]
            means_by_group = tf.reduce_mean(inputs_by_group, axis=[1,2,4], keep_dims=True)
            zeromean_inputs_by_group = inputs_by_group - means_by_group
            stddev_by_group = tf.sqrt(1e-6 + tf.reduce_mean(tf.square(zeromean_inputs_by_group), axis=[1,2,4], keep_dims=True))
            standardized_inputs_by_group = zeromean_inputs_by_group / stddev_by_group

            broadcast_shape = [1, 1, 1] + [num_groups, group_size]
            loc = tf.reshape(loc, broadcast_shape) # broadcast for dynamic batch size
            scale = tf.reshape(scale, broadcast_shape)
            groupnormed_inputs_by_group = scale * standardized_inputs_by_group + loc

            target_shape = [-1] + inputs.get_shape().as_list()[1:]
            groupnormed_inputs = tf.reshape(groupnormed_inputs_by_group, target_shape) # shape [b,h,w,gf] where the i-th set of f consecutive entries on channel axis will correspond to group i.
            return groupnormed_inputs

    def residual_block(self, inputs, training, name):
        ## residual block without bottleneck
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)

            conv1 = tf.layers.conv2d(
                inputs, filters=self.num_filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_init, activation=None)
            norm1 = self.group_norm(conv1, num_groups=(self.num_filters // 4), name='gn1')

            act1 = self.activation(norm1)
            drop1 = tf.layers.dropout(act1, rate=self.res_block_internal_dropout_rate, training=training)

            conv2 = tf.layers.conv2d(
                drop1, filters=self.num_filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_init, activation=None)
            norm2 = self.group_norm(conv2, num_groups=(self.num_filters // 4), name='gn2')

            act2 = self.activation(inputs + norm2)
            output = act2

            return output

    def decoder_block(self, inputs, training, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            glorot_init = tf.glorot_normal_initializer()

            res_blocks = [inputs]
            for j in range(0, self.res_blocks_per_decoder_block):
                res_block = self.residual_block(res_blocks[j], training=training, name='res_block_' + str(j+1))
                res_blocks.append(res_block)

            last = res_blocks[self.res_blocks_per_decoder_block]

            dconv = tf.layers.conv2d_transpose(
                last, filters=self.num_filters, kernel_size=4, strides=2, padding='same',
                kernel_initializer=glorot_init, activation=None)
            output = dconv

            return output

    def apply_sr(self, Mu_z_perceptual, Mu_z_mem, training=True):
        # apply the novel stochastic regularizer.
        assert self.sr_delta == 2 * self.sr_epsilon and self.sr_epsilon >= np.finfo(np.float32).eps
        alpha_vec = self.sr_alpha * tf.ones_like(Mu_z_perceptual)
        beta_vec = self.sr_beta * tf.ones_like(Mu_z_perceptual)

        s_dist = tfp.distributions.Beta(concentration1=alpha_vec, concentration0=beta_vec) # this is Beta(alpha, beta), see tfp docs.
        s = s_dist.sample()
        left = self.sr_gamma - self.sr_epsilon
        t = left + self.sr_delta * s
        Mu_z = t * Mu_z_perceptual + (1. - t) * Mu_z_mem
        return Mu_z

    def pz_given_wMb(self, w, M, b, pz_logsigma):
        with tf.variable_scope('pz_given_wM', reuse=tf.AUTO_REUSE):
            mu = tf.einsum('bkc,bk->bc', M, w) + b
            logsigma = pz_logsigma * tf.ones_like(mu)
            z_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
            return z_dist

    def pw(self, batch_size):
        with tf.variable_scope('pw', reuse=tf.AUTO_REUSE):
            mu = tf.zeros(dtype=tf.float32, shape=[batch_size, self.memory_size])
            logsigma = tf.zeros(dtype=tf.float32, shape=[batch_size, self.memory_size])
            w_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(logsigma))
            return w_dist

    def compute_good_init_tf(self, episode_zs):
        # this implements the k-means++ initialization algorithm (Arthur and Vassilvitskii, 2007)
        # within tensorflow rather than require two session.run calls. 
        # we do not run Lloyd's iterations, only the seeding algorithm.
        batch_size = tf.shape(episode_zs)[0]
        good_init_idxs_onehot_array = tf.TensorArray(dtype=tf.float32, size=self.num_clusters, infer_shape=True)
        uniform_probs = (1.0/tf.cast(float(self.episode_len), dtype=tf.float32)) * tf.ones(dtype=tf.float32, shape=[batch_size, self.episode_len])
        uniform_onehot = tfp.distributions.OneHotCategorical(probs=uniform_probs, dtype=tf.float32)
        first = uniform_onehot.sample()
        good_init_idxs_onehot_array = good_init_idxs_onehot_array.write(0, first)
        min_squared_distances_from_selecteds = tf.reduce_sum(tf.square(tf.expand_dims(tf.einsum('bt,btc->bc', first, episode_zs), 1) - episode_zs), axis=-1) # bt

        loop_init_vars = (1, min_squared_distances_from_selecteds, good_init_idxs_onehot_array)
        loop_cond = lambda i, *_: i < self.num_clusters
        def loop_body(i, min_squared_distances_from_selecteds, array):
            probs = min_squared_distances_from_selecteds / tf.reduce_sum(min_squared_distances_from_selecteds, axis=-1, keep_dims=True) # shape: [B, T]
            distribution = tfp.distributions.OneHotCategorical(probs=probs, dtype=tf.float32)
            selected_onehot = distribution.sample()
            selected_code = tf.einsum('bt,btc->bc', selected_onehot, episode_zs)
            squared_distances_from_selected_code = tf.reduce_sum(tf.square(tf.expand_dims(selected_code, 1) - episode_zs), axis=-1) # bt
            min_squared_distances_from_selecteds_new = tf.minimum(min_squared_distances_from_selecteds, squared_distances_from_selected_code)
            return (i+1, min_squared_distances_from_selecteds_new, array.write(i, selected_onehot))

        _, _, good_init_idxs_onehot_array_final = tf.while_loop(loop_cond, loop_body, loop_init_vars)
        good_init_idxs_onehot_final = tf.transpose(good_init_idxs_onehot_array_final.stack(), perm=[1,0,2]) # [b,h,t]
        result = tf.stop_gradient(good_init_idxs_onehot_final)
        return result

    def compute_good_init_tf_tree(self, episode_zs_g12):
        # this implements the k-means++ initialization algorithm (Arthur and Vassilvitskii, 2007)
        # within tensorflow, rather than require multiple session.run calls to fetch the codes and then feed the initializations through placeholder variables.
        # we do not run Lloyd's iterations, only the seeding algorithm.
        #
        # this is the variant designed for tree-structured memory, and it obtains an init for q(b_{1:H}^(g)) for g=1,2,...,G, associated with each of the code segment/hop.
        # in this method, the inits are computed separately, and in parallel, for each hop.
        #
        # note that episode_zs_g12 should have shape [B,T,G,C/G].
        batch_size = tf.shape(episode_zs_g12)[0]
        episode_zs_g12_hopmajor = tf.transpose(episode_zs_g12, perm=[0,2,1,3]) # [b,g,t,c/g]. it's batch major, but the time axis is later than the hop axis.
        good_init_idxs_onehot_array = tf.TensorArray(dtype=tf.float32, size=self.num_clusters, infer_shape=True) # we will be returning the one-hot indices.

        uniform_probs = (1.0/tf.cast(float(self.episode_len), dtype=tf.float32)) * tf.ones(dtype=tf.float32, shape=[batch_size, self.num_hops, self.episode_len])
        uniform_onehot = tfp.distributions.OneHotCategorical(probs=uniform_probs, dtype=tf.float32)
        first_onehot = uniform_onehot.sample()

        good_init_idxs_onehot_array = good_init_idxs_onehot_array.write(0, first_onehot) # save.

        # accumulators for each item in the episode; what is its min squared euclidean distance to the codes selected so far? 
        # in the tree-structured model, the clustering inits, and hence the accumulators, are computed separately for each code segment idx g=1,2,...,G.
        first_selected_code = tf.einsum('bgt,bgtc->bgc', first_onehot, episode_zs_g12_hopmajor) # [b,g,c/g].
        min_squared_distances_from_selecteds = tf.reduce_sum(tf.square(tf.expand_dims(first_selected_code, 2) - episode_zs_g12_hopmajor), axis=-1) # [b,g,t].
        
        loop_init_vars = (1, min_squared_distances_from_selecteds, good_init_idxs_onehot_array)
        loop_cond = lambda i, *_: i < self.num_clusters
        def loop_body(i, min_squared_distances_from_selecteds, array):
            probs = min_squared_distances_from_selecteds / tf.reduce_sum(min_squared_distances_from_selecteds, axis=-1, keep_dims=True) # shape: [B, G, T]
            distribution = tfp.distributions.OneHotCategorical(probs=probs, dtype=tf.float32)
            selected_onehot = distribution.sample() # bgt
            selected_code = tf.einsum('bgt,bgtc->bgc', selected_onehot, episode_zs_g12_hopmajor) # [b,g,c/g].
            squared_distances_from_selected_code = tf.reduce_sum(tf.square(tf.expand_dims(selected_code, 2) - episode_zs_g12_hopmajor), axis=-1) # bgt
            min_squared_distances_from_selecteds_new = tf.minimum(min_squared_distances_from_selecteds, squared_distances_from_selected_code) # bgt
            return (i+1, min_squared_distances_from_selecteds_new, array.write(i, selected_onehot))

        _, _, good_init_idxs_onehot_array_final = tf.while_loop(loop_cond, loop_body, loop_init_vars)
        good_init_idxs_onehot_final_stacked = good_init_idxs_onehot_array_final.stack() # [h,b,g,t], since we wrote H items to the array, each with shape bgt.
        good_init_idxs_onehot_final = tf.transpose(good_init_idxs_onehot_final_stacked, perm=[1,2,0,3]) # [b,g,h,t]
        result = tf.stop_gradient(good_init_idxs_onehot_final)
        return result

    def train(self, sess, input_xs):
        feed_dict = {
            self.input_xs: input_xs,
            self.is_train: True
        }
        fetch_list = [
            self.train_op, 
            self.elbo_per_frame, 
            self.global_step, 
            self.merged_summaries
        ]
        _, elbo_per_frame, step, summaries = sess.run(fetch_list, feed_dict=feed_dict)
        return elbo_per_frame, step, summaries

    def evaluate(self, sess, input_xs):
        feed_dict = {
            self.input_xs: input_xs,
            self.is_train: False
        }
        elbo_per_frame = sess.run(self.elbo_per_frame, feed_dict=feed_dict)
        return elbo_per_frame

    def get_prior_memory_state(self, sess, input_xs):
        # uses input_xs only to get a batch size in get_initial_state method for p(M).
        feed_dict = {
            self.input_xs: input_xs,
            self.is_train: False
        }
        R, U, Mu_b, Sigma_b = sess.run([self.pMb.R, self.pMb.U, self.pMb.Mu_b, self.pMb.Sigma_b], feed_dict=feed_dict)
        return MemoryState(R=R, U=U, Mu_b=Mu_b, Sigma_b=Sigma_b)

    def get_posterior_memory_state(self, sess, input_xs):
        # this uses input_xs to compute variational posterior q(Omega), and to return either empirical probabilities over mixture assignment sequences, 
        # or simply a list of the mixture assignment sequences; for the purposes of this method, we take the argmax of q(s_t^(g)) in order to heuristically determine 
        # which hard assignment sequences occurred in the episode; we could also average over soft assignment probabilities, but this is more expensive (it would be O(H^G) time and space).
        feed_dict = {
            self.input_xs: input_xs,
            self.is_train: False
        }
        fetch_list = [
            self.qMb.R, 
            self.qMb.U, 
            self.qMb.Mu_b, 
            self.qMb.Sigma_b, 
            #self.s1_probs, 
            #self.s2_probs_given_s1
            self.sg_sharps
        ]
        #R, U, Mu_b, Sigma_b, s1_probs, s2_probs_given_s1 = sess.run(fetch_list, feed_dict=feed_dict)
        R, U, Mu_b, Sigma_b, sg_sharps = sess.run(fetch_list, feed_dict=feed_dict)
        return MemoryState(R=R, U=U, Mu_b=Mu_b, Sigma_b=Sigma_b), sg_sharps #s1_probs, s2_probs_given_s1

    def generate_from_memory_state(self, sess, memory_state, sg_sharps): #s1_probs, s2_probs_given_s1):
        feed_dict = {
            self.input_R: memory_state.R,
            self.input_U: memory_state.U,
            self.input_Mu_b: memory_state.Mu_b,
            self.input_Sigma_b: memory_state.Sigma_b,
            #self.input_s1_probs: s1_probs,
            #self.input_s2_probs_given_s1: s2_probs_given_s1,
            self.input_sg_sharps: sg_sharps,
            self.is_train: False
        }
        generated_x = sess.run(self.generated_x, feed_dict=feed_dict)
        return generated_x

    def read_from_memory_state(self, sess, memory_state, x):
        feed_dict = {
            self.input_R: memory_state.R,
            self.input_U: memory_state.U,
            self.input_Mu_b: memory_state.Mu_b,
            self.input_Sigma_b: memory_state.Sigma_b,
            self.query_x: x,
            self.is_train: False
        }
        read_x, read_z = sess.run([self.read_x, self.read_z], feed_dict=feed_dict)
        return read_x, read_z

    def copy(self, sess, x):
        feed_dict = {
            self.query_x: x,
            self.is_train: False
        }
        copied_x, copied_z = sess.run([self.copied_x, self.copied_z], feed_dict=feed_dict)
        return copied_x, copied_z
