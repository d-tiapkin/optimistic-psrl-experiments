from rlberry.utils.jit_setup import numba_jit
import numpy as np


@numba_jit
def backward_induction_in_place(Q, V, R, P, horizon, gamma=1.0, vmax=np.inf):
    """
    Backward induction to compute Q and V functions in
    the finite horizon setting.
    Takes as input the arrays where to store Q and V.
    Parameters
    ----------
    Q:  numpy.ndarray
        array of shape (horizon, S, A) where to store the Q function
    V:  numpy.ndarray
        array of shape (horizon, S+1) where to store the V function
    R : numpy.ndarray
        array of shape (S, A, B) contaning the rewards, where S is the number
        of states and A is the number of actions Bnumber of bootstraps
    P : numpy.ndarray
        array of shape (S, A, S+1,B) such that P[s,a,ns,b] is the probability of
        arriving at ns by taking action a in state s for the bootstrap b
    horizon : int
        problem horizon
    gamma : double
        discount factor, default = 1.0
    vmax : double
        maximum possible value in V
        default = np.inf
    """
    S, A, B = R.shape
    next_S = P.shape[-2]
    for hh in range(horizon - 1, -1, -1):
        for ss in range(S):
            max_q = -np.inf
            for aa in range(A):
                q_aa = -np.inf
                for bb in range(B):
                    _q_aa_b = R[ss, aa, bb]
                    if hh < horizon - 1:
                        for ns in range(next_S):
                            _q_aa_b += gamma * P[ss, aa, ns, bb] * V[hh + 1, ns]
                    if _q_aa_b > q_aa:
                        q_aa = _q_aa_b
                if q_aa > max_q:
                    max_q = q_aa
                Q[hh, ss, aa] = q_aa
            V[hh, ss] = max_q
            if V[hh, ss] > vmax:
                V[hh, ss] = vmax


@numba_jit
def backward_induction_sd(Q, V, R, P, gamma=1.0, vmax=np.inf):
    """
    In-place implementation of backward induction to compute Q and V functions
    in the finite horizon setting.
    Assumes R and P are stage-dependent.
    Parameters
    ----------
    Q:  numpy.ndarray
        array of shape (H, S, A) where to store the Q function
    V:  numpy.ndarray
        array of shape (H, S+1) where to store the V function
    R : numpy.ndarray
        array of shape (H, S, A,B) contaning the rewards, where S is the number
        of states and A is the number of actions
    P : numpy.ndarray
        array of shape (H, S, A, S+1,B) such that P[h, s, a, ns,b] is the probability of
        arriving at ns by taking action a in state s at stage h in bootstrap b.
    gamma : double, default: 1.0
        discount factor
    vmax : double, default: np.inf
        maximum possible value in V
    """
    H, S, A, B= R.shape
    S_next = P.shape[-2]
    _q_aa = np.zeros(B)
    for hh in range(H - 1, -1, -1):
        for ss in range(S):
            max_q = -np.inf
            for aa in range(A):
                q_aa = -np.inf
                for bb in range(B):
                    _q_aa_b = R[hh, ss, aa, bb]
                    if hh < H - 1:
                        for ns in range(S_next):
                            _q_aa_b += gamma * P[hh, ss, aa, ns,bb] * V[hh + 1, ns]
                    if _q_aa_b > q_aa:
                        q_aa = _q_aa_b
                if q_aa > max_q:
                    max_q = q_aa
                Q[hh, ss, aa] = q_aa
            V[hh, ss] = max_q
            # clip V
            if V[hh, ss] > vmax:
                V[hh, ss] = vmax