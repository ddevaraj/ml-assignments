from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(O_t = x_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    # TODO
    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(O_t = x_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - delta: (num_state*L) A numpy array delta[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S, L])
        ###################################################
        # Edit here
        print(self.A, "hello", self.B, "pi", self.pi)
        obs = [self.obs_dict[i] for i in Osequence]
        print(obs, self.obs_dict, Osequence)
        for state in range(S):
            delta[state, 0] = self.pi[state] * self.B[state][obs[0]]
        for t in range(1,L):
            for state in range(S):
                delta[state][t] = self.B[state][obs[t]] * sum([self.A[i][state] * delta[i][t-1] for i in range(S)])
        ###################################################
        return delta

    # TODO:
    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(O_t = x_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array gamma[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        gamma = np.zeros([S, L])
        ###################################################
        # Edit here
        obs = [self.obs_dict[i] for i in Osequence]
        for state in range(S):
            gamma[state, L-1] = 1
        for t in range(L-2, -1, -1):
            for state in range(S):
                gamma[state, t] = sum([self.A[state, i] * self.B[i, obs[t+1]] * gamma[i,t+1] for i in range(S)])
        ###################################################
        return gamma

    # TODO:
    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        delta = self.forward(Osequence)
        prob = sum(delta[:,-1])
        ###################################################
        return prob

    # TODO:
    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i | O, λ)
        """
        prob = 0
        ###################################################
        # Edit here
        gamma = self.backward(Osequence)
        delta = self.forward(Osequence)
        px = self.sequence_prob(Osequence)
        # prob = [[gamma[state, i] * delta[state, i] for i in range(len(Osequence))]for state in range(len(self.pi))]
        prob = gamma * delta
        prob = np.divide(prob, px)

        ###################################################
        return prob

    # TODO:
    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Edit here
        obs = [self.obs_dict[i] for i in Osequence]
        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S, L])
        paths = np.zeros([S, L], dtype="int")

        for s in range(S):
            delta[s, 0] = self.pi[s] * self.B[s, obs[0]]
            paths[s, 0] = 0

        for t in range(1, L):
            for s in range(S):
                delta_s = [delta[s_prime, t - 1] * self.A[s_prime, s] for s_prime in
                          range(S)]
                # print('deltas', delta_s)
                delta[s, t] = max(delta_s) * self.B[s, obs[t]]
                # print('delta',delta)
                paths[s, t] = np.argmax(delta_s)
        # print('paths', paths)
        last_state = np.argmax(delta[:,-1])
        path.append(last_state)
        # print('path', path)
        for t in range(L-2,-1,-1):
            path.append(paths[path[-1], t+1])

        # print(self.state_dict, path)
        final_path = []
        def get_key(val):
            for key, value in self.state_dict.items():
                if val == value:
                    return key
        for index in path:
            final_path.append(get_key(index))
        # print(final_path)
        ###################################################
        return final_path[::-1]
        # return path