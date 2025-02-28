import torch
import torch.nn as nn


class HumanFeedbackRewardModel(nn.Module):
    def __init__(self, state_dimension, compute_device):
        """
        Reward model trained from human preferences for LunarLander-v3.
        
        Parameters
        ----------
        state_dimension : int
            Dimension of the state input vector
        compute_device : str
            Device to use for computation ('cpu' or 'cuda')
        """
        super().__init__()

        # Configuration
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.device = compute_device
        self.state_dim = state_dimension
        self.ensemble_size = 3
        self.hidden_dims = [128, 64]
        self.dropout_rate = 0.5

        # Model components
        self.ensemble_networks = []
        self.ensemble_optimizers = []
        self.loss_history = []

        # Initialize the ensemble
        self._initialize_ensemble()

    def _build_reward_network(self):
        """
        Create a single reward prediction network.
        
        Returns
        -------
        nn.Sequential
            Neural network for reward prediction
        """
        network = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dims[1], 1),
        )

        return network

    def get_ensemble_networks(self):
        """
        Get the list of networks in the ensemble.
        
        Returns
        -------
        list
            List of reward prediction networks
        """
        return self.ensemble_networks

    def _initialize_ensemble(self):
        """Initialize ensemble of reward networks with their optimizers."""
        for _ in range(self.ensemble_size):
            network = self._build_reward_network().to(self.device)
            optimizer = torch.optim.Adam(
                network.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )

            self.ensemble_networks.append(network)
            self.ensemble_optimizers.append(optimizer)

    def forward(self, state, train=False):
        """
        Forward pass through all networks in the ensemble.
        
        Parameters
        ----------
        state : torch.Tensor or numpy.ndarray
            Input state or batch of states
        train : bool
            Whether to use training mode or evaluation mode
            
        Returns
        -------
        list
            List of reward predictions from each network in the ensemble
        """
        # Ensure input is a tensor on the correct device
        state_tensor = torch.Tensor(state).to(self.device)
        reward_predictions = []

        for network in self.ensemble_networks:
            if train:
                network.train()
                reward_predictions.append(network(state_tensor))
            else:
                network.eval()
                with torch.no_grad():
                    reward_predictions.append(network(state_tensor))

        return reward_predictions

    def update(self, rewards_segment_a, rewards_segment_b, human_preferences):
        """
        Update the reward networks based on human preference feedback.
        
        Parameters
        ----------
        rewards_segment_a : torch.Tensor
            Predicted rewards for segment A
        rewards_segment_b : torch.Tensor
            Predicted rewards for segment B
        human_preferences : torch.Tensor
            Human preference labels (one-hot encoded)
        """
        batch_size = rewards_segment_a.shape[0]
        
        # Process each batch element
        for batch_idx in range(batch_size):
            # Initialize loss for this batch element
            batch_loss = 0
            
            # Calculate preference probabilities and loss for each data point
            for reward_a, reward_b, preference in zip(
                rewards_segment_a[batch_idx], 
                rewards_segment_b[batch_idx], 
                human_preferences
            ):
                # Ensure tensors are on the correct device
                reward_a = reward_a.to(self.device)
                reward_b = reward_b.to(self.device)
                preference = preference.to(self.device)
                
                # Calculate Bradley-Terry probability model
                prob_prefer_a = torch.exp(reward_a) / (torch.exp(reward_a) + torch.exp(reward_b))
                prob_prefer_b = torch.exp(reward_b) / (torch.exp(reward_a) + torch.exp(reward_b))
                
                # Negative log-likelihood loss
                batch_loss -= preference[0] * prob_prefer_a + preference[1] * prob_prefer_b
            
            # Create gradient variable
            loss_variable = torch.autograd.Variable(batch_loss, requires_grad=True).to(self.device)

        # Zero gradients for all optimizers
        for optimizer in self.ensemble_optimizers:
            optimizer.zero_grad()
        
        # Backpropagate loss
        loss_variable.backward()
        
        # Update all networks
        for optimizer in self.ensemble_optimizers:
            optimizer.step()
        
        # Store loss for monitoring
        self.loss_history.append(loss_variable.item())