import matplotlib.pyplot as plt

class ProgressPlotter:
    """
    Class for plotting the progress of rewards over steps using matplotlib.

    It maintains internal lists of steps, total rewards, and individual progress rewards,
    and updates the plot dynamically.
    """
    def __init__(self):
        # Set up the figure and axes
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title('Reward Progress')
        self.ax.set_xlabel('Steps')
        self.ax.set_ylabel('Value')

        # Initialize lists to store step counts and rewards
        self.steps = []
        self.rewards = []        # Stores individual rewards as lists
        self.total_rewards = []  # Stores total rewards

    def update_plot(self, step_count, progress_reward, total_reward):
        """
        Update the plot with new data points.

        :param step_count: Current step number
        :param progress_reward: Dictionary of individual reward components keyed by name
        :param total_reward: Total reward value at current step
        """
        # Append new data to the internal lists
        self.steps.append(step_count)
        self.total_rewards.append(total_reward)
        self.rewards.append(list(progress_reward.values()))  # Save individual rewards as list

        # Clear the axes to redraw
        self.ax.clear()
        self.ax.set_title('Reward Progress')
        self.ax.set_xlabel('Steps')
        self.ax.set_ylabel('Value')

        # Plot total rewards in blue
        self.ax.plot(self.steps, self.total_rewards, label="Total Reward", color="blue")
        
        # Plot each individual reward over steps with separate lines
        for idx, key in enumerate(progress_reward.keys()):
            # Extract reward values for this key across all recorded steps
            reward_values = [r[idx] for r in self.rewards]
            self.ax.plot(self.steps, reward_values, label=key)

        # Add legend to identify lines
        self.ax.legend()

        # Refresh the plot display with a small pause for animation effect
        plt.draw()
        plt.pause(0.1)
