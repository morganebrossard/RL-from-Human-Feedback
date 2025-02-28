from random import sample
import torch
from reward import HumanFeedbackRewardModel


class HumanRewardModelTrainer:
    def __init__(self, preference_database, state_dimension, compute_device):
        """
        Trainer for the human reward model in LunarLander-v3.
        
        Parameters
        ----------
        preference_database : list
            Dataset of preference triplets (state_a, state_b, human_preference)
        state_dimension : int
            Dimension of the state space
        compute_device : str
            Device to use for computation ('cpu' or 'cuda')
        """
        self.reward_model = HumanFeedbackRewardModel(state_dimension, compute_device)
        self.preference_database = preference_database  # Memory of (state_a, state_b, human_preference) triplets
        self.batch_size = 10  # Sample size for training iterations

    def train_human_rewarder(self):
        """
        Train the reward model on a random batch from the preference database.
        
        Returns
        -------
        HumanFeedbackRewardModel
            The updated reward model
        """
        # Sample batch_size triplets from preference database
        sampled_triplets = sample(self.preference_database, self.batch_size)

        # Initialize lists to store batch data
        predicted_rewards_a = []
        predicted_rewards_b = []
        human_preferences = []

        # Process each triplet
        for state_a, state_b, preference in sampled_triplets:
            human_preferences.append(preference)

            # Convert states to tensors
            state_a_tensor = torch.tensor(state_a, dtype=torch.float32, device=self.reward_model.device)
            state_b_tensor = torch.tensor(state_b, dtype=torch.float32, device=self.reward_model.device)

            # Get reward predictions from the ensemble
            reward_predictions_a = self.reward_model(state_a_tensor, train=True)
            reward_predictions_b = self.reward_model(state_b_tensor, train=True)

            # Sum rewards across ensemble members
            predicted_rewards_a.append(torch.stack(reward_predictions_a).sum(dim=0))
            predicted_rewards_b.append(torch.stack(reward_predictions_b).sum(dim=0))

        # Convert lists to tensors for batch processing
        predicted_rewards_a_tensor = torch.stack(predicted_rewards_a)
        predicted_rewards_b_tensor = torch.stack(predicted_rewards_b)
        human_preferences_tensor = torch.tensor(
            human_preferences, 
            dtype=torch.float32, 
            device=self.reward_model.device
        )

        # Update reward model with human preferences
        self.reward_model.update(
            predicted_rewards_a_tensor, 
            predicted_rewards_b_tensor, 
            human_preferences_tensor
        )
        
        return self.reward_model

    def pretrain_human_rewarder(self):
        """
        Pretrain the reward model before starting the reinforcement learning process.
        
        Returns
        -------
        HumanFeedbackRewardModel
            The pretrained reward model
        """
        total_epochs = 500
        reporting_interval = 50
        
        for epoch in range(total_epochs):
            self.reward_model = self.train_human_rewarder()
            
            if epoch % reporting_interval == 0:
                print(f"Pretraining reward model: {epoch}/{total_epochs}")

        return self.reward_model