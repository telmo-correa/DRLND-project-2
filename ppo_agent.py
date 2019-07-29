import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model import GaussianActorCriticNet
from collections import deque


class Memory:
    """Helper class for managing observed experiences."""

    def __init__(self, device, memory_size):
        """Initializes the data structure.

        :param device:  (torch.device) Object representing the device where to allocate tensors
        :param memory_size: (int) Maximum capacity of memory buffer
        """
        self.device = device
        self.elements = deque(maxlen=memory_size)

    def add(self, states, actions, log_probs, values, rewards, non_terminals):
        """Adds a set of experience tuples for each agent to the memory.

        :param states:  (list of states)  States observed for each agent
        :param actions:  (list of actions)  Actions selected for each agent
        :param log_probs:  (tensor)  Log probabilities for each action selection
        :param values:  (tensor)  Value function for each agent
        :param rewards:  (list of rewards)  Rewards obtained for each agent
        :param non_terminals:  (numpy array)  List of flags for non-terminal states (1 - done)
        """
        p_states = torch.tensor(states).float().to(self.device)
        p_actions = torch.tensor(actions).float().to(self.device)
        p_log_probs = log_probs
        p_values = values
        p_rewards = torch.tensor(rewards).float().to(self.device)
        p_non_terminals = torch.from_numpy(non_terminals).float().to(self.device)

        self.elements.append((p_states, p_actions, p_log_probs, p_values, p_rewards, p_non_terminals))

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.elements)

    def clear(self):
        """Removes all elements from internal memory."""
        self.elements.clear()

    def reverse_iterator(self):
        """Return a reversed iterator over experiences."""
        return reversed(self.elements)


class MultiAgent:
    """Class responsible for coordinating multiple parallel agents using PPO."""

    def __init__(
        self,
        state_size,
        action_size,
        shared_network_units,
        actor_network_units,
        critic_network_units,
        optimizer_learning_rate=5e-4,
        optimizer_epsilon=1e-5,
        trajectory_length=1000,
        gamma=0.99,
        gae_lambda=0.9,
        optimization_steps=16,
        batch_size=256,
        gradient_clip=0.25,
        ppo_ratio_clip_epsilon=0.2,
        entropy_penalty_weight=0.01,
        value_loss_weight=1.0,
        std_init=0,
        std_function=None,
        device=None
    ):
        """


        :param state_size:
        :param action_size:
        :param shared_network_units:
        :param actor_network_units:
        :param critic_network_units:
        :param optimizer_learning_rate:
        :param optimizer_epsilon:
        :param trajectory_length:
        :param gamma:
        :param gae_lambda:
        :param optimization_steps:
        :param batch_size:
        :param gradient_clip:
        :param ppo_ratio_clip_epsilon:
        :param entropy_penalty_weight:
        :param value_loss_weight:
        :param device:
        """
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Problem dimensions
        self.device = device
        self.state_size = state_size
        self.action_size = action_size

        # Network
        self.network = GaussianActorCriticNet(
            state_size=state_size,
            action_size=action_size,
            shared_layers=shared_network_units,
            actor_layers=actor_network_units,
            critic_layers=critic_network_units,
            std_init=std_init,
            std_function=std_function
        )
        self.network.to(device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=optimizer_learning_rate,
            eps=optimizer_epsilon
        )

        # Memory
        self.memory = Memory(device=device, memory_size=trajectory_length)

        # Hyperparameters
        self.trajectory_length = trajectory_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.optimization_steps = optimization_steps
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip
        self.ppo_ratio_clip_epsilon = ppo_ratio_clip_epsilon
        self.entropy_penalty_weight = entropy_penalty_weight
        self.value_loss_weight = value_loss_weight

        # Internal state
        self.t_step = 0
        self.max_std = 1
        self.last_act = {}

    def act(self, states):
        """Evaluates an action for each agent using the current policy, given a list of initial
        states.

        :param states:  (list of states) Initial states for each agent
        :return: (list of actions) Actions selected for each agent
        """
        stacked_states = torch.from_numpy(np.stack(states)).float().to(self.device)
        network_output = self.network(stacked_states)

        self.last_act['states'] = states
        self.last_act['network_output'] = network_output

        return network_output['action'].cpu().detach().numpy()

    def step(self, states, actions, rewards, next_states, dones):
        """Memorize a single environment step for all agents -- and perform trajectory rollback
        and learning if enough steps have elapsed.

        This class assumes on-policy learning -- this method should be called in sequence, once for
        each experience across all agents.

        :param states:  (list of states) Initial states for each agent
        :param actions:  (list of actions) Actions selected for each agent
        :param rewards:  (list) Rewards obtained for each agent
        :param next_states:  (list of states) Subsequent states for each agent
        :param dones:  (list) Boolean flags for terminal states
        :return:
        """
        if 'states' in self.last_act and np.all(self.last_act['states'] == states):
            network_output = self.last_act['network_output']
            self.last_act.clear()
        else:
            stacked_states = torch.from_numpy(np.stack(states)).float().to(self.device)
            stacked_actions = torch.from_numpy(np.stack(actions)).float().to(self.device)
            network_output = self.network(stacked_states, stacked_actions)

        self.memory.add(
            states=states,
            actions=actions,
            log_probs=network_output['log_prob'].detach(),
            values=network_output['v'].detach(),
            rewards=rewards,
            non_terminals=1 - np.array(dones)
        )

        self.t_step = (self.t_step + 1) % self.trajectory_length
        if self.t_step == 0:
            trajectories = self.collect_trajectories(next_states)
            self.learn(trajectories)
            self.memory.clear()

    def collect_trajectories(self, next_states):
        """Collect a set of trajectories for each agent.

        :param next_states:  Final state to start rollback from
        :return:  (dict) Dictionary containing tensors for states, actions,
            log_probs, returns, and advantages
        """
        stacked_next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        returns = self.network(stacked_next_states)['v'].detach()

        p_next_values = returns
        advantages = torch.zeros(len(next_states)).float().to(self.device)

        # Initialize trajectories as deque
        trajectories = {}
        for k in ['states', 'actions', 'log_probs', 'returns', 'advantages']:
            trajectories[k] = deque()

        # Perform rollback on trajectories
        for tuple in self.memory.reverse_iterator():
            p_states, p_actions, p_log_probs, p_values, p_rewards, p_non_terminals = tuple

            returns = p_rewards * self.gamma * p_non_terminals * returns
            td_error = p_rewards + self.gamma * p_non_terminals * p_next_values - p_values
            advantages = advantages * self.gae_lambda * self.gamma * p_non_terminals + td_error

            trajectories['states'].appendleft(p_states)
            trajectories['actions'].appendleft(p_actions)
            trajectories['log_probs'].appendleft(p_log_probs)
            trajectories['returns'].appendleft(returns)
            trajectories['advantages'].appendleft(advantages)

            p_next_values = p_values

        # Translate deques into stacked tensors
        for k in ['states', 'actions', 'log_probs', 'returns', 'advantages']:
            trajectories[k] = torch.stack(list(trajectories[k]))

        # Normalize advantages
        adv = trajectories['advantages']
        trajectories['advantages'] = (adv - adv.mean()) / adv.std()

        return trajectories

    def learn(self, trajectories):
        """Perform optimization steps on network over samples from the provided trajectory.

        :param trajectories:  (dict) Dictionary containing tensors for states, actions,
            log_probs, returns, and advantages
        """
        trajectory_size = trajectories['states'].shape[0]

        # For each optimization step, collect a random batch of indexes and propagate loss
        for _ in range(self.optimization_steps):
            batch_indices = torch.randint(low=0, high=trajectory_size, size=(self.batch_size,)).long()

            sampled_states = trajectories['states'][batch_indices].detach()
            sampled_actions = trajectories['actions'][batch_indices].detach()
            sampled_log_probs = trajectories['log_probs'][batch_indices].detach()
            sampled_returns = trajectories['returns'][batch_indices].detach()
            sampled_advantages = trajectories['advantages'][batch_indices].detach()

            L = self.loss(sampled_states, sampled_actions, sampled_log_probs, sampled_returns, sampled_advantages)

            self.optimizer.zero_grad()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
            L.backward()
            self.optimizer.step()

            del L

    def loss(self, states, actions, log_probs, rewards, advantages):
        """Compute the loss function for a sampled batch of experiences

        :param states:  (tensor) Sampled states
        :param actions:  (tensor) Sampled actions
        :param log_probs:  (tensor) Sampled log probabilities
        :param rewards:  (tensor) Sampled propagated rewards
        :param advantages:  (tensor) Sampled propagated advantages
        :return:  Loss function for optimizer
        """

        # Compute predictions from states, actions to get value for critic loss and log probabilities for entropy
        predictions = self.network(states, actions)

        # Ratio is product of probability ratios -- store as log probabilities instead
        ratio = (predictions['log_prob'] - log_probs).sum(-1).exp()

        # Compute clipped policy loss (L_clip)
        loss_original = ratio * advantages
        loss_clipped = ratio.clamp(1 - self.ppo_ratio_clip_epsilon, 1 + self.ppo_ratio_clip_epsilon) * advantages
        policy_loss = -torch.min(loss_original, loss_clipped).mean()

        # Apply penalty for entropy
        entropy_penalty = -self.entropy_penalty_weight * predictions['entropy'].mean()

        # Compute value function loss (mean squared error)
        value_loss = self.value_loss_weight * F.mse_loss(predictions['v'].view(-1), rewards.view(-1))

        return policy_loss + entropy_penalty + value_loss

    def save(self, filename):
        """Saves the policy network to a file.

        :param filename:  Filename where to save the policy network
        """
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        """Loads the policy network from a file.

        :param filename: Filename from where to load the policy network
        """
        self.network.load_state_dict(torch.load(filename))
