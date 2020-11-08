import mxnet as mx

from mxnet.ndarray import linalg_gemm as gemm
from mxnet.ndarray import linalg_gemm2 as gemm2
from mxnet.ndarray import linalg_potrf as potrf
from mxnet.ndarray import linalg_trsm as trsm
from mxnet.ndarray import linalg_sumlogdiag as sumlogdiag
import mxnet.ndarray as nd
import numpy as np



def ISSM(z, b, F, a, g, sigma, m_prior, S_prior):
    '''
    The documentation for this code can be found in :
    https://gluon.mxnet.io/chapter12_time-series/issm-scratch.html
    '''

    H = F.shape[0] # dim of latent state
    T = z.shape[0] # num of observations

    eye_h = nd.array(np.eye(H))

    mu_seq = []
    S_seq = []
    log_p_seq = []

    for t in range(T):

        if t == 0:
            # At the first time step, use the prior
            mu_h = m_prior
            S_hh = S_prior
        else:
            # Otherwise compute using update eqns.
            F_t = F[:, :, t]
            g_t = g[:, t].reshape((H,1))

            mu_h = gemm2(F_t, mu_t)
            S_hh = gemm2(F_t, gemm2(S_t, F_t, transpose_b=1)) + \
                   gemm2(g_t, g_t, transpose_b=1)

        a_t = a[:, t].reshape((H,1))
        mu_v = gemm2(mu_h, a_t, transpose_a=1)

        # Compute the Kalman gain (vector)
        S_hh_x_a_t = gemm2(S_hh, a_t)

        sigma_t = sigma[t]
        S_vv = gemm2(a_t, S_hh_x_a_t, transpose_a=1) + nd.square(sigma_t)
        kalman_gain = nd.broadcast_div(S_hh_x_a_t, S_vv)

        # Compute the error (delta)
        delta = z[t] - b[t] - mu_v

        # Filtered estimates
        mu_t = mu_h + gemm2(kalman_gain, delta)

        # Joseph's symmetrized update for covariance:
        ImKa = nd.broadcast_sub(eye_h, gemm2(kalman_gain, a_t, transpose_b=1))
        S_t = gemm2(gemm2(ImKa, S_hh), ImKa, transpose_b=1) + \
                nd.broadcast_mul(gemm2(kalman_gain, kalman_gain, transpose_b=1), nd.square(sigma_t))

        # likelihood term
        log_p = (-0.5 * (delta * delta / S_vv
                         + np.log(2.0 * np.pi)
                         + nd.log(S_vv))
                 )

        mu_seq.append(mu_t)
        S_seq.append(S_t)
        log_p_seq.append(log_p)


    return log_p_seq
    #return mu_seq, S_seq, log_p_seq
