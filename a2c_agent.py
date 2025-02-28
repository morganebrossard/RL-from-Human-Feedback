import torch
import torch.optim as optim
import random
from torch.optim.lr_scheduler import LinearLR


class AdvantageActorCriticAgent:
    def __init__(self, state_dim, action_dim, random_seed, compute_device, discount_factor, 
                 actor_learning_rate, critic_learning_rate, update_frequency, actor_model, critic_model):
        """Initialize an Agent object.
        
        Parameters
        ----------
        state_dim (tuple): Dimension of each state
        action_dim (int): Dimension of each action
        random_seed (int): Random seed
        compute_device (str): Device to use (CPU or GPU)
        discount_factor (float): Discount factor
        actor_learning_rate (float): Actor learning rate
        critic_learning_rate (float): Critic learning rate 
        update_frequency (int): How often to update the network
        actor_model (Model): PyTorch Actor Model
        critic_model (Model): PyTorch Critic Model
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = random.seed(random_seed)
        self.device = compute_device
        self.discount_factor = discount_factor
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.update_frequency = update_frequency
        self.timestep_counter = 0
        
        # Constants
        self.CRITIC_LOSS_COEF = 0.5
        self.ENTROPY_COEF = 0.001
        self.MAX_TRAINING_ITERATIONS = 16e7

        # Initialize Actor Network
        self.actor_network = actor_model(state_dim[0], action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(
            self.actor_network.parameters(), 
            lr=self.actor_learning_rate
        )
        self.actor_scheduler = LinearLR(
            self.actor_optimizer, 
            start_factor=1, 
            end_factor=0, 
            total_iters=self.MAX_TRAINING_ITERATIONS
        )

        # Initialize Critic Network
        self.critic_network = critic_model(state_dim[0]).to(self.device)
        self.critic_optimizer = optim.Adam(
            self.critic_network.parameters(), 
            lr=self.critic_learning_rate
        )
        self.critic_scheduler = LinearLR(
            self.critic_optimizer, 
            start_factor=1, 
            end_factor=0, 
            total_iters=self.MAX_TRAINING_ITERATIONS
        )

        # Initialize Memory Buffers
        self.reset_memory()

    def step(self, state, log_prob, entropy, reward, done, next_state):
        """Store experience and learn if it's time to update.
        
        Parameters
        ----------
        state: Current state
        log_prob: Log probability of chosen action
        entropy: Entropy of action distribution
        reward: Reward received
        done: Whether episode has terminated
        next_state: Next state
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_value = self.critic_network(state_tensor)

        # Save experience in memory
        self.log_probs.append(log_prob)
        self.values.append(state_value)
        reward_value = reward if reward is not None else 0.0
        self.rewards.append(torch.tensor([float(reward_value)], dtype=torch.float32, device=self.device))
        self.masks.append(torch.tensor([1 - done], dtype=torch.float32, device=self.device))
        self.entropies.append(entropy)

        self.timestep_counter = (self.timestep_counter + 1) % self.update_frequency

        # Update if it's time
        if self.timestep_counter == 0:
            self.learn(next_state)
            self.reset_memory()

    def act(self, state):
        """Returns action, log_prob, entropy for given state according to current policy.
        
        Parameters
        ----------
        state: Current state
        
        Returns
        -------
        action: Selected action
        log_prob: Log probability of selected action
        entropy: Entropy of action distribution
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_distribution = self.actor_network(state_tensor)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy().mean()

        return action.item(), log_prob, entropy

    def learn(self, next_state):
        """Update policy and value parameters using Monte-Carlo estimates.
        
        Parameters
        ----------
        next_state: Next state for bootstrapping
        """
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_value = self.critic_network(next_state_tensor)

        # Calculate returns
        returns = self.compute_returns(next_value, self.discount_factor)

        # Prepare tensors
        log_probs_tensor = torch.cat(self.log_probs)
        returns_tensor = torch.cat(returns).detach()
        values_tensor = torch.cat(self.values)

        # Calculate advantage
        advantage = returns_tensor - values_tensor

        # Calculate losses
        actor_loss = -(log_probs_tensor * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        entropy_term = sum(self.entropies)
        
        # Combined loss
        total_loss = (
            actor_loss + 
            self.CRITIC_LOSS_COEF * critic_loss - 
            self.ENTROPY_COEF * entropy_term
        )

        # Perform optimization
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        # Update learning rate schedulers
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def reset_memory(self):
        """Clear memory buffers."""
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.entropies = []

    def compute_returns(self, next_value, gamma=0.99):
        """Compute returns using bootstrap method.
        
        Parameters
        ----------
        next_value: Value of the next state
        gamma: Discount factor
        
        Returns
        -------
        List of discounted returns
        """
        bootstrap_value = next_value
        returns = []
        
        for step in reversed(range(len(self.rewards))):
            bootstrap_value = self.rewards[step] + gamma * bootstrap_value * self.masks[step]
            returns.insert(0, bootstrap_value)
            
        return returns