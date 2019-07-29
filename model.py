import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import chain


class GaussianActorCriticNet(nn.Module):

    def __init__(self, state_size, action_size, shared_layers, actor_layers, critic_layers, std_init=0, std_function=None):
        super(GaussianActorCriticNet, self).__init__()

        self.shared_network = GaussianActorCriticNet._create_shared_network(state_size, shared_layers)
        shared_output_size = state_size if len(shared_layers) == 0 else shared_layers[-1]

        self.actor_network = GaussianActorCriticNet._create_actor_network(shared_output_size, actor_layers, action_size)
        self.critic_network = GaussianActorCriticNet._create_critic_network(shared_output_size, critic_layers)

        self.std = nn.Parameter(torch.ones(action_size) * std_init)
        if std_function is None:
            self.std_function = F.softplus
        else:
            self.std_function = std_function


    @staticmethod
    def _create_shared_network(state_size, shared_layers):
        iterator = chain([state_size], shared_layers)

        last_size = None
        args = []
        for layer_size in iterator:
            if last_size is not None:
                args.append(nn.Linear(last_size, layer_size))
                args.append(nn.ReLU())
            last_size = layer_size

        return nn.Sequential(*args)

    @staticmethod
    def _create_actor_network(input_size, actor_layers, action_size):
        iterator = chain([input_size], actor_layers, [action_size])

        last_size = None
        args = []
        for layer_size in iterator:
            if last_size is not None:
                args.append(nn.Linear(last_size, layer_size))
                args.append(nn.ReLU())
            last_size = layer_size

        # Replace last ReLU layer with tanh
        del args[-1]
        args.append(nn.Tanh())

        return nn.Sequential(*args)

    @staticmethod
    def _create_critic_network(input_size, critic_layers):
        iterator = chain([input_size], critic_layers, [1])

        last_size = None
        args = []
        for layer_size in iterator:
            if last_size is not None:
                args.append(nn.Linear(last_size, layer_size))
                args.append(nn.ReLU())
            last_size = layer_size

        # Remove last ReLU layer
        del args[-1]

        return nn.Sequential(*args)

    def forward(self, states, action=None):
        shared_output = self.shared_network(states)
        mu = self.actor_network(shared_output)
        value = self.critic_network(shared_output)

        distribution = torch.distributions.Normal(mu, self.std_function(self.std))
        if action is None:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy().sum(-1).unsqueeze(-1)

        return {
            'v': value.squeeze(1),
            'action': action,
            'log_prob': log_prob,
            'entropy': entropy
        }
