import numpy as np

import gym.spaces as spaces
from rlberry.agents import AgentWithSimplePolicy
from rlberry.exploration_tools.discrete_counter import DiscreteCounter
from algorithms.opsrl.utils import (
    backward_induction_in_place, 
    backward_induction_sd,
) 

from rlberry.agents.dynprog.utils import backward_induction_in_place as backward_induction_in_place_o
from rlberry.agents.dynprog.utils import backward_induction_sd as backward_induction_sd_o


import logging
logger = logging.getLogger(__name__)

class OPSRLAgent(AgentWithSimplePolicy):
    """
    OPSRL algorithm from ours with beta prior for the "Bernoullized" rewards
    (instead of Gaussian-gamma prior).
    Notes
    -----
    The recommended policy after all the episodes is computed without
    exploration bonuses.
    Parameters
    ----------
    env : gym.Env
        Environment with discrete states and actions.
    gamma : double, default: 1.0
        Discount factor in [0, 1]. If gamma is 1.0, the problem is set to
        be finite-horizon.
    horizon : int
        Horizon of the objective function. If None and gamma<1, set to
        1/(1-gamma).
    scale_prior_reward : double, delfault: 1.0
        scale of the Beta (uniform) prior,
        i.e prior is Beta(scale_prior_reward*(1,1))
    thompson_samples: int, default: 1
        number of thompson samples
    prior_transition : string, default: 'uniform'
        type of Dirichlet prior in {'optimistic', 'uniform'}.
    bernoullized_reward: bool, default: True
        If true the rewards are Bernoullized
    reward_free : bool, default: False
        If true, ignores rewards and uses only 1/n bonuses.
    stage_dependent : bool, default: False
        If true, assume that transitions and rewards can change with the stage h.
    References
    ----------
    .. [1] Osband et al., 2013
        (More) Efficient Reinforcement Learning via Posterior Sampling
        https://arxiv.org/abs/c.0940
    """

    name = "OPSRL"

    def __init__(
        self,
        env,
        gamma=1.0,
        horizon=100,
        bernoullized_reward=True,
        scale_prior_reward=1.0,
        thompson_samples=1,
        prior_transition='uniform',
        scale_prior_transition=None,
        reward_free=False,
        stage_dependent=False,
        **kwargs
    ):
        # init base class
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        self.gamma = gamma
        self.horizon = horizon
        self.bernoullized_reward = bernoullized_reward
        self.scale_prior_reward = scale_prior_reward
        self.thompson_samples = thompson_samples
        assert prior_transition in ['uniform','optimistic']
        self.prior_transition = prior_transition
        self.scale_prior_transition = scale_prior_transition
        if scale_prior_transition is None:
            if self.prior_transition == 'uniform':
                self.scale_prior_transition = 1.0 / self.env.observation_space.n
            else:
                self.scale_prior_transition = 1.0
        self.reward_free = reward_free
        self.stage_dependent = stage_dependent

        # check environment
        assert isinstance(self.env.observation_space, spaces.Discrete)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # other checks
        assert gamma >= 0 and gamma <= 1.0
        if self.horizon is None:
            assert gamma < 1.0, "If no horizon is given, gamma must be smaller than 1."
            self.horizon = int(np.ceil(1.0 / (1.0 - gamma)))

        # maximum value
        r_range = self.env.reward_range[1] - self.env.reward_range[0]
        if r_range == np.inf or r_range == 0.0:
            logger.warning(
                "{}: Reward range is  zero or infinity. ".format(self.name)
                + "Setting it to 1."
            )
            r_range = 1.0

        self.v_max = np.zeros(self.horizon)
        self.v_max[-1] = r_range
        for hh in reversed(range(self.horizon - 1)):
            self.v_max[hh] = r_range + self.gamma * self.v_max[hh + 1]

        # initialize
        self.reset()

    def reset(self, **kwargs):
        H = self.horizon
        S = self.env.observation_space.n
        A = self.env.action_space.n

        if self.stage_dependent:
            shape_hsa = (H, S, A)
            shape_hsas = (H, S, A, S)
            if self.prior_transition == 'optimistic':
                shape_hsas = (H, S, A, S+1)
        else:
            shape_hsa = (S, A)
            shape_hsas = (S, A, S)
            if self.prior_transition == 'optimistic':
                shape_hsas = (S, A, S+1)



        # Prior transitions
        self.N_sas = self.scale_prior_transition * np.ones(shape_hsas)
        if self.prior_transition == 'optimistic':
            self.N_sas = np.zeros(shape_hsas)
            self.N_sas[...,-1] += self.scale_prior_transition

        # Prior rewards
        self.M_sa = self.scale_prior_reward * np.ones(shape_hsa + (2,))

        # Value functions
        self.V = np.zeros((H, S))
        if self.prior_transition == 'optimistic':
            self.V = np.zeros((H, S+1))
        self.Q = np.zeros((H, S, A))

        # Init V if needed
        if self.prior_transition == 'optimistic':
            for hh in range(self.horizon):
                self.V[hh,:] = self.v_max[hh]

        # for rec. policy
        self.V_policy = np.zeros((H, S))
        self.Q_policy = np.zeros((H, S, A))

        # ep counter
        self.episode = 0

        # useful object to compute total number of visited states & entropy of visited states
        self.counter = DiscreteCounter(
            self.env.observation_space, self.env.action_space
        )

    def policy(self, observation):
        state = observation
        assert self.Q_policy is not None
        return self.Q_policy[0, state, :].argmax()

    def _get_action(self, state, hh=0):
        """Sampling policy."""
        assert self.Q is not None
        return self.Q[hh, state, :].argmax()

    def _update(self, state, action, next_state, reward, hh):
        bern_reward = reward
        if self.bernoullized_reward:
            bern_reward = self.rng.binomial(1, reward)
        # update posterior
        if self.stage_dependent:
            self.N_sas[hh, state, action, next_state] += 1
            self.M_sa[hh, state, action, 0] += bern_reward
            self.M_sa[hh, state, action, 1] += 1 - bern_reward

        else:
            self.N_sas[state, action, next_state] += 1
            self.M_sa[state, action, 0] += bern_reward
            self.M_sa[state, action, 1] += 1 - bern_reward

    def _run_episode(self):
        # sample reward and transitions from posterior
        B = self.thompson_samples
        M_sab_zero = np.repeat(self.M_sa[..., 0, np.newaxis], B, -1)
        M_sab_one = np.repeat(self.M_sa[..., 1, np.newaxis], B, -1)
        N_sasb = np.repeat(self.N_sas[...,np.newaxis], B, axis=-1)
        self.R_samples = self.rng.beta(M_sab_zero, M_sab_one)
        self.P_samples = self.rng.gamma(N_sasb)
        self.P_samples = self.P_samples / self.P_samples.sum(-2, keepdims=True)
        # run backward induction
        if self.stage_dependent:
            backward_induction_sd(
                self.Q, self.V, self.R_samples, self.P_samples, self.gamma, self.v_max[0]
            )
        else:
            backward_induction_in_place(
                self.Q,
                self.V,
                self.R_samples,
                self.P_samples,
                self.horizon,
                self.gamma,
                self.v_max[0],
            )
        # interact for H steps
        episode_rewards = 0
        state = self.env.reset()
        for hh in range(self.horizon):
            action = self._get_action(state, hh)
            next_state, reward, done, _ = self.env.step(action)
            episode_rewards += reward  # used for logging only

            self.counter.update(state, action)

            if self.reward_free:
                reward = 0.0  # set to zero before update if reward_free

            self._update(state, action, next_state, reward, hh)

            state = next_state
            if done:
                break

        # update info
        self.episode += 1

        # writer
        if self.writer is not None:
            self.writer.add_scalar("episode_rewards", episode_rewards, self.episode)
            self.writer.add_scalar(
                "n_visited_states", self.counter.get_n_visited_states(), self.episode
            )

        # return sum of rewards collected in the episode
        return episode_rewards

    def fit(self, budget: int, **kwargs):
        """
        Train the agent using the provided environment.
        Parameters
        ----------
        budget: int
            number of episodes. Each episode runs for self.horizon unless it
            enconters a terminal state in which case it stops early.
        """
        del kwargs
        n_episodes_to_run = budget
        count = 0
        while count < n_episodes_to_run:
            self._run_episode()
            count += 1

        # compute Q function for the recommended policy
        R_hat = self.M_sa[..., 0] / (self.M_sa[..., 0] + self.M_sa[..., 1])
        P_hat = self.N_sas / self.N_sas.sum(-1, keepdims=True)
        if self.stage_dependent:
            backward_induction_sd_o(
                self.Q_policy, self.V_policy, R_hat, P_hat, self.gamma, self.v_max[0]
            )
        else:
            backward_induction_in_place_o(
                self.Q_policy,
                self.V_policy,
                R_hat,
                P_hat,
                self.horizon,
                self.gamma,
                self.v_max[0],
            )