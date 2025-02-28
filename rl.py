import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output
import gymnasium as gym
from a2c_agent import AdvantageActorCriticAgent
from space import Space
from reward import HumanFeedbackRewardModel
from human_reward import HumanRewardModelTrainer


class ReinforcementLearningFromHumanPreferences:
    def __init__(self, state_shape, random_seed, compute_device, discount_factor, 
                 actor_lr, critic_lr, update_frequency, actor_network, critic_network, 
                 use_real_human=False):
        """Initialize system with environment, agent and reward model.
        
        Parameters
        ----------
        state_shape : tuple
            Dimension of the state space
        random_seed : int
            Seed for reproducibility
        compute_device : str
            Device to use for computation ('cpu' or 'cuda')
        discount_factor : float
            Discount factor for future rewards
        actor_lr : float
            Learning rate for the actor network
        critic_lr : float
            Learning rate for the critic network
        update_frequency : int
            How often to update the networks
        actor_network : torch.nn.Module
            Actor network architecture
        critic_network : torch.nn.Module
            Critic network architecture
        use_real_human : bool
            Whether to use real human feedback (vs. simulated)
        """
        self.compute_device = compute_device
        self.use_real_human = use_real_human

        # Initialize the environment
        self.environment = gym.make("LunarLander-v3")
        self.action_count = self.environment.action_space.n

        # Initialize the agent
        self.agent = AdvantageActorCriticAgent(
            state_dim=state_shape,
            action_dim=self.action_count,
            random_seed=random_seed,
            compute_device=compute_device,
            discount_factor=discount_factor,
            actor_learning_rate=actor_lr,
            critic_learning_rate=critic_lr,
            update_frequency=update_frequency,
            actor_model=actor_network,
            critic_model=critic_network,
        )

        # Initialize preference datastore for human feedback
        self.class_space = Space(env=self.environment, agent=self.agent, real_human_check=use_real_human)
        self.space = self.class_space.space

        # Initialize reward model based on human feedback
        self.reward_model = HumanFeedbackRewardModel(state_shape[0], compute_device)

    def train_from_human_preferences(self, episode_count=500):
        """Train agent using human preference feedback.
        
        Parameters
        ----------
        episode_count : int
            Number of episodes to train for
            
        Returns
        -------
        list
            Scores achieved in each episode
        """
        # Initialize and pretrain the reward model
        reward_model_trainer = HumanRewardModelTrainer(
            self.space, 
            self.agent.state_dim[0], 
            self.compute_device
        )
        self.reward_model = reward_model_trainer.pretrain_human_rewarder()

        # Monitoring variables
        total_timesteps = 1
        true_rewards = []
        predicted_rewards = []
        timestep_records = []
        state_buffer = []
        true_reward_buffer = []
        
        # Training loop
        for episode_idx in range(episode_count):
            # Reset the environment
            current_state, _ = self.environment.reset()
            current_state = np.ascontiguousarray(current_state, dtype=np.float32)

            episode_true_reward = 0
            episode_predicted_reward = 0
            episode_step = 0

            # Episode loop
            while True:
                # Update the reward model periodically
                if episode_step % 10 == 0:
                    reward_model_trainer.train_human_rewarder()

                # Select an action
                action, action_log_prob, action_entropy = self.agent.act(current_state)
                
                # Execute the action
                next_state, env_reward, is_terminal, is_truncated, _ = self.environment.step(action)

                # Compute predicted reward from the reward model
                state_tensor = torch.Tensor(current_state).unsqueeze(0).to(self.reward_model.device)
                model_reward = torch.stack(self.reward_model(state_tensor)).mean().cpu().detach().numpy()

                # Update cumulative rewards
                episode_predicted_reward += model_reward
                episode_true_reward += env_reward

                # Process the next state
                next_state = np.ascontiguousarray(next_state, dtype=np.float32)

                # Update the agent using predicted reward
                self.agent.step(
                    current_state,
                    action_log_prob,
                    action_entropy,
                    model_reward,
                    is_terminal or is_truncated,
                    next_state,
                )

                # Update state for next iteration
                current_state = next_state
                
                # Store observations and rewards for preference learning
                state_buffer.append(current_state)
                true_reward_buffer.append(env_reward)

                # Manage buffer size
                if len(state_buffer) > 50:
                    state_buffer.pop(0)
                    true_reward_buffer.pop(0)

                # Update preference database when enough data is collected
                if len(state_buffer) == 50:
                    mid_point = 25
                    self.space = self.class_space.feeding_space(
                        state_buffer[:mid_point],
                        state_buffer[mid_point:],
                        real_reward_1=np.sum(true_reward_buffer[:mid_point]),
                        real_reward_2=np.sum(true_reward_buffer[mid_point:]),
                        total_steps=20
                    )

                    # Clear buffers after updating preference database
                    state_buffer = []
                    true_reward_buffer = []

                # Increment step counters
                episode_step += 1
                total_timesteps += 1
                
                # Check if episode is complete
                if is_terminal or is_truncated:
                    break

            # Record episode results
            true_rewards.append(episode_true_reward)
            predicted_rewards.append(episode_predicted_reward)
            timestep_records.append(total_timesteps)

            # Visualize training progress
            clear_output(True)
            plt.title("LunarLander-v3")
            plt.plot(timestep_records, true_rewards, label="True reward")
            plt.plot(timestep_records, predicted_rewards, label="Predicted reward")
            plt.ylabel("Reward")
            plt.xlabel("Timestep")
            plt.legend(loc=4)
            plt.show()

            print(f"Episode: {episode_idx}, Score: {episode_true_reward}")

        return true_rewards