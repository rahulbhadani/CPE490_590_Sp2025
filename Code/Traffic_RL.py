import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from collections import deque
import math
import pygame
import sys
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class IDMVehicle:
    """Intelligent Driver Model vehicle implementation"""
    
    def __init__(self, v0=30.0, T=1.5, a=1.0, b=1.5, delta=4.0, s0=2.0, position=0.0, velocity=0.0, vehicle_length=5.0):
        # IDM parameters
        self.v0 = v0  # desired velocity in m/s
        self.T = T    # safe time headway in s
        self.a = a    # maximum acceleration in m/s^2
        self.b = b    # comfortable deceleration in m/s^2
        self.delta = delta  # acceleration exponent
        self.s0 = s0  # minimum gap in m
        
        # Vehicle state
        self.position = position  # position in m
        self.velocity = velocity  # velocity in m/s
        self.length = vehicle_length  # vehicle length in m
        
        # History for plotting
        self.position_history = [position]
        self.velocity_history = [velocity]
        self.acceleration_history = [0.0]
        self.time_history = [0.0]
        
        # Color for PyGame visualization (default blue for IDM vehicles)
        self.color = (0, 0, 255)  # RGB: Blue
        
    def step(self, lead_vehicle, dt):
        """Update position and velocity based on IDM model"""
        # Calculate gap to leading vehicle
        if lead_vehicle:
            s = lead_vehicle.position - self.position - self.length
            s = max(0.1, s)  # Ensure minimum gap
            delta_v = self.velocity - lead_vehicle.velocity
        else:
            # If no lead vehicle, assume free road
            s = float('inf')
            delta_v = 0
        
        # Desired minimum gap
        s_star = self.s0 + max(0, self.velocity * self.T + 
                           (self.velocity * delta_v) / (2 * np.sqrt(self.a * self.b)))
        
        # Calculate acceleration
        acceleration = self.a * (1 - (self.velocity / self.v0) ** self.delta - (s_star / s) ** 2)
        
        # Apply acceleration limits
        acceleration = max(-3 * self.b, min(self.a, acceleration))  # Limit deceleration/acceleration
        
        # Update velocity and position using acceleration
        self.velocity = max(0, self.velocity + acceleration * dt)
        self.position += self.velocity * dt + 0.5 * acceleration * dt ** 2
        
        # Store history
        self.position_history.append(self.position)
        self.velocity_history.append(self.velocity)
        self.acceleration_history.append(acceleration)
        self.time_history.append(self.time_history[-1] + dt)
        
        return acceleration
    
    def calculate_fuel_consumption(self, acceleration):
        """
        Calculate fuel consumption based on the VT-Micro model
        Returns fuel consumption in ml/s
        """
        # Simplified VT-Micro model parameters for a mid-size vehicle
        v = self.velocity * 3.6  # convert to km/h
        a = acceleration  # m/s^2
        
        # Coefficients for the model (simplified)
        if a >= 0:  # Acceleration mode
            fuel_rate = 0.444 + 0.09 * v + 0.05 * a * v
        else:  # Deceleration mode
            fuel_rate = max(0.444 + 0.09 * v + 0.02 * a * v, 0.3)  # Idle fuel rate is about 0.3 ml/s
            
        return fuel_rate

class ReplayBuffer:
    """Experience replay buffer for the RL agent"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    """Deep Q-Network for the RL agent"""
    
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class RLAgent:
    """Reinforcement Learning agent based on DQN"""
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_space = np.linspace(-3.0, 2.0, action_dim)  # acceleration space from -3 m/s^2 to 2 m/s^2
        
        # Q-Networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # History for plotting
        self.position = 0.0
        self.velocity = 0.0
        self.position_history = [0.0]
        self.velocity_history = [0.0]
        self.acceleration_history = [0.0]
        self.time_history = [0.0]
        self.rewards_history = []
        self.length = 5.0  # vehicle length in m
        
        # Color for PyGame visualization (red for RL agent)
        self.color = (255, 0, 0)  # RGB: Red
        
    def get_state(self, lead_vehicles, lane_length):
        """Create state representation for the agent"""
        # Get information about the 3 vehicles ahead
        states = []
        
        # Find the 3 vehicles ahead of the RL agent
        vehicles_ahead = []
        for vehicle in lead_vehicles:
            if vehicle.position > self.position:
                vehicles_ahead.append(vehicle)
        
        # Sort by position (closest first)
        vehicles_ahead.sort(key=lambda v: v.position)
        
        # Take up to 3 vehicles ahead
        vehicles_ahead = vehicles_ahead[:3]
        
        # Add self state
        states.append(self.velocity / 30.0)  # Normalized velocity
        states.append((lane_length - self.position) / lane_length)  # Normalized position
        
        # Add information about vehicles ahead
        for i in range(3):
            if i < len(vehicles_ahead):
                vehicle = vehicles_ahead[i]
                # Relative position (normalized)
                states.append((vehicle.position - self.position) / 100.0)
                # Relative velocity (normalized)
                states.append((vehicle.velocity - self.velocity) / 30.0)
            else:
                # If no vehicle, use placeholder values
                states.append(1.0)  # Far ahead
                states.append(0.0)  # Same velocity
        
        return np.array(states)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(q_values).item()
    
    def train(self, batch_size=64):
        """Train the Q-network using experience replay"""
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert actions to indices
        action_indices = actions.long()
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Update target network parameters"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def step(self, action_idx, lead_vehicle, dt):
        """Update position and velocity based on selected action"""
        acceleration = self.action_space[action_idx]
        
        # Apply acceleration limits
        if lead_vehicle:
            # Simple safety mechanism
            safe_distance = self.velocity * 1.5 + 2.0  # Simple safe distance calculation
            actual_distance = lead_vehicle.position - self.position - self.length
            
            if actual_distance < safe_distance and acceleration > 0:
                # If too close and accelerating, override with deceleration
                acceleration = min(acceleration, -1.0)
        
        # Update velocity and position
        self.velocity = max(0, self.velocity + acceleration * dt)
        self.position += self.velocity * dt + 0.5 * acceleration * dt ** 2
        
        # Store history
        self.position_history.append(self.position)
        self.velocity_history.append(self.velocity)
        self.acceleration_history.append(acceleration)
        self.time_history.append(self.time_history[-1] + dt)
        
        return acceleration
    
    def calculate_fuel_consumption(self, acceleration):
        """
        Calculate fuel consumption based on the VT-Micro model
        Returns fuel consumption in ml/s
        """
        # Similar to the IDM vehicle fuel calculation
        v = self.velocity * 3.6  # convert to km/h
        a = acceleration  # m/s^2
        
        # Coefficients for the model (simplified)
        if a >= 0:  # Acceleration mode
            fuel_rate = 0.444 + 0.09 * v + 0.05 * a * v
        else:  # Deceleration mode
            fuel_rate = max(0.444 + 0.09 * v + 0.02 * a * v, 0.3)  # Idle fuel rate is about 0.3 ml/s
            
        return fuel_rate
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class TrafficSimulationRenderer:
    """PyGame-based renderer for traffic simulation"""
    
    def __init__(self, width=1200, height=400, scale_factor=0.8, lane_length=1000):
        self.width = width
        self.height = height
        self.scale_factor = scale_factor  # Scale factor to convert meters to pixels
        self.lane_length = lane_length
        
        # Initialize PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Traffic Simulation with RL Agent")
        self.clock = pygame.time.Clock()
        
        # Fonts for displaying information
        self.font = pygame.font.SysFont('Arial', 16)
        self.large_font = pygame.font.SysFont('Arial', 24)
        
        # Colors
        self.bg_color = (230, 230, 230)  # Light gray
        self.road_color = (80, 80, 80)   # Dark gray
        self.lane_mark_color = (255, 255, 255)  # White
        self.text_color = (0, 0, 0)      # Black
        
        # Vehicle dimensions in pixels
        self.vehicle_width = 20
        self.vehicle_height = 40
        
    def _world_to_screen(self, world_x, world_y=0):
        """Convert world coordinates to screen coordinates"""
        screen_x = int(world_x * self.scale_factor) % self.width
        screen_y = self.height // 2 - world_y - self.vehicle_height // 2
        return screen_x, screen_y
    
    def render(self, vehicles, rl_agent, episode=0, step=0, reward=0, total_fuel=0, show_info=True):
        """Render the current state of the simulation"""
        # Handle PyGame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Clear the screen
        self.screen.fill(self.bg_color)
        
        # Draw road
        road_height = 60
        pygame.draw.rect(self.screen, self.road_color, (0, self.height // 2 - road_height // 2, self.width, road_height))
        
        # Draw lane markings
        line_length = 20
        line_gap = 30
        y = self.height // 2
        for x in range(0, self.width, line_length + line_gap):
            pygame.draw.line(self.screen, self.lane_mark_color, (x, y), (x + line_length, y), 2)
        
        # Draw IDM vehicles
        for vehicle in vehicles:
            # Get screen coordinates
            screen_x, screen_y = self._world_to_screen(vehicle.position)
            
            # Create vehicle rectangle
            vehicle_rect = pygame.Rect(screen_x, screen_y, self.vehicle_width, self.vehicle_height)
            
            # Draw vehicle
            pygame.draw.rect(self.screen, vehicle.color, vehicle_rect)
            
            # Draw velocity indicator (as a line extending from the vehicle, length proportional to velocity)
            vel_line_length = min(vehicle.velocity * 1.5, 60)  # Limit the line length
            pygame.draw.line(self.screen, 
                             (0, 150, 150), 
                             (screen_x + self.vehicle_width // 2, screen_y + self.vehicle_height), 
                             (screen_x + self.vehicle_width // 2, screen_y + self.vehicle_height + vel_line_length), 
                             2)
        
        # Draw RL agent
        if rl_agent:
            # Get screen coordinates
            screen_x, screen_y = self._world_to_screen(rl_agent.position)
            
            # Create vehicle rectangle
            agent_rect = pygame.Rect(screen_x, screen_y, self.vehicle_width, self.vehicle_height)
            
            # Draw vehicle with distinct appearance
            pygame.draw.rect(self.screen, rl_agent.color, agent_rect)
            
            # Draw outline to make RL agent more distinct
            pygame.draw.rect(self.screen, (255, 255, 0), agent_rect, 2)  # Yellow outline
            
            # Draw velocity indicator
            vel_line_length = min(rl_agent.velocity * 1.5, 60)  # Limit the line length
            pygame.draw.line(self.screen, 
                             (150, 0, 150), 
                             (screen_x + self.vehicle_width // 2, screen_y + self.vehicle_height), 
                             (screen_x + self.vehicle_width // 2, screen_y + self.vehicle_height + vel_line_length), 
                             2)
        
        # Display information
        if show_info:
            info_texts = [
                f"Episode: {episode}",
                f"Step: {step}",
                f"Reward: {reward:.2f}",
                f"Epsilon: {rl_agent.epsilon:.4f}",
                f"RL Agent Velocity: {rl_agent.velocity:.2f} m/s",
                f"Total Fuel: {total_fuel:.2f} ml"
            ]
            
            for i, text in enumerate(info_texts):
                text_surface = self.font.render(text, True, self.text_color)
                self.screen.blit(text_surface, (10, 10 + i * 20))
            
            # Legend
            legend_texts = [
                "RL Agent (Red)",
                "IDM Vehicles (Blue)"
            ]
            
            for i, text in enumerate(legend_texts):
                text_surface = self.font.render(text, True, self.text_color)
                self.screen.blit(text_surface, (self.width - 200, 10 + i * 20))
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        self.clock.tick(30)  # 30 FPS
    
    def close(self):
        """Close the renderer"""
        pygame.quit()

def run_simulation_with_visualization(num_episodes=100, episode_steps=500, num_vehicles=10, lane_length=1000, save_interval=10, render=True, render_interval=1):
    """Run the traffic simulation with IDM vehicles, RL agent, and visualization"""
    # Define state and action dimensions
    state_dim = 8  # self velocity, self position, 3 vehicles ahead (distance + relative velocity)
    action_dim = 10  # Different acceleration values
    
    # Create RL agent
    agent = RLAgent(state_dim, action_dim)
    
    # Create renderer if needed
    renderer = None
    if render:
        renderer = TrafficSimulationRenderer(lane_length=lane_length)
    
    # Training loop
    episode_rewards = []
    best_reward = -float('inf')
    dt = 0.1  # time step in seconds
    
    for episode in range(num_episodes):
        # Reset environment
        vehicles = []
        
        # Create IDM vehicles with randomized initial positions
        for i in range(num_vehicles):
            # Randomize starting positions (spaced apart with some randomness)
            pos = i * 15 + np.random.uniform(-2, 2)
            # Randomize starting velocities
            vel = np.random.uniform(20, 25)
            # Add IDM parameters with some randomness to create more realistic behavior
            idm_params = {
                'v0': np.random.uniform(25, 30),  # desired velocity
                'T': np.random.uniform(1.2, 1.8),  # safe time headway
                'a': np.random.uniform(0.8, 1.2),  # max acceleration
                'b': np.random.uniform(1.2, 1.8),  # comfortable deceleration
                'delta': 4.0,
                's0': np.random.uniform(1.5, 2.5),  # minimum gap
                'position': pos,
                'velocity': vel
            }
            vehicles.append(IDMVehicle(**idm_params))
        
        # Reset RL agent
        agent.position = 0.0
        agent.velocity = np.random.uniform(20, 25)
        agent.position_history = [agent.position]
        agent.velocity_history = [agent.velocity]
        agent.acceleration_history = [0.0]
        agent.time_history = [0.0]
        
        # Episode variables
        episode_reward = 0
        total_fuel = 0
        
        # Simulation loop
        for step in range(episode_steps):
            # Sort vehicles by position (descending)
            vehicles.sort(key=lambda v: v.position, reverse=True)
            
            # Update IDM vehicles
            for i, vehicle in enumerate(vehicles):
                lead_vehicle = vehicles[i-1] if i > 0 else None
                acceleration = vehicle.step(lead_vehicle, dt)
                total_fuel += vehicle.calculate_fuel_consumption(acceleration) * dt
            
            # Find the vehicle directly ahead of the RL agent
            lead_vehicle = None
            for vehicle in vehicles:
                if vehicle.position > agent.position:
                    if lead_vehicle is None or vehicle.position < lead_vehicle.position:
                        lead_vehicle = vehicle
            
            # Get state, select action, and update RL agent
            state = agent.get_state(vehicles, lane_length)
            action_idx = agent.select_action(state)
            acceleration = agent.step(action_idx, lead_vehicle, dt)
            
            # Calculate fuel consumption for RL agent
            agent_fuel = agent.calculate_fuel_consumption(acceleration) * dt
            total_fuel += agent_fuel
            
            # Calculate next state
            next_state = agent.get_state(vehicles, lane_length)
            
            # Calculate reward
            # Reward smooth driving, avoid crashes, maintain good speed, reduce fuel consumption
            fuel_consumption = agent.calculate_fuel_consumption(acceleration)
            
            # Check if too close to the vehicle ahead
            too_close = False
            if lead_vehicle:
                gap = lead_vehicle.position - agent.position - agent.length
                if gap < 5:  # 5 meters is too close
                    too_close = True
            
            # Reward components
            velocity_reward = agent.velocity / 30.0  # Encourage maintaining speed
            smoothness_reward = -abs(acceleration) / 3.0  # Discourage harsh acceleration/deceleration
            fuel_reward = -fuel_consumption / 5.0  # Reduce fuel consumption
            safety_penalty = -10.0 if too_close else 0.0  # Heavy penalty for unsafe distance
            
            # Combined reward
            reward = velocity_reward + smoothness_reward + fuel_reward + safety_penalty
            
            # Check if done (end of lane or crash)
            done = agent.position >= lane_length or too_close
            
            # Store transition in replay buffer
            agent.replay_buffer.add(state, action_idx, reward, next_state, done)
            
            # Train the agent
            agent.train()
            
            # Update target network periodically
            if step % 100 == 0:
                agent.update_target_network()
            
            # Update total reward
            episode_reward += reward
            
            # Render if needed
            if render and step % render_interval == 0:
                renderer.render(vehicles, agent, episode=episode, step=step, 
                               reward=reward, total_fuel=total_fuel)
            
            # Break if done
            if done:
                break
        
        # Store episode reward
        agent.rewards_history.append(episode_reward)
        episode_rewards.append(episode_reward)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model('best_traffic_rl_model.pth')
        
        # Print episode information
        if (episode + 1) % save_interval == 0:
            avg_reward = np.mean(episode_rewards[-save_interval:])
            print(f"Episode: {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    print("Training completed!")
    agent.save_model('final_traffic_rl_model.pth')
    
    # Close renderer if needed
    if renderer:
        renderer.close()
    
    # Return trained agent and final set of vehicles for evaluation
    return agent, vehicles

def evaluate_traffic_flow_with_visualization(agent, num_vehicles=10, lane_length=1000, steps=500, render=True, render_interval=1):
    """Evaluate the trained RL agent's impact on traffic flow with visualization"""
    # Create vehicles for evaluation
    vehicles = []
    
    # Create IDM vehicles with similar initial positions
    for i in range(num_vehicles):
        pos = i * 15 + np.random.uniform(-2, 2)
        vel = np.random.uniform(20, 25)
        idm_params = {
            'v0': np.random.uniform(25, 30),
            'T': np.random.uniform(1.2, 1.8),
            'a': np.random.uniform(0.8, 1.2),
            'b': np.random.uniform(1.2, 1.8),
            'delta': 4.0,
            's0': np.random.uniform(1.5, 2.5),
            'position': pos, 
            'velocity': vel
        }
        vehicles.append(IDMVehicle(**idm_params))
    
    # Reset RL agent
    agent.position = 0.0
    agent.velocity = np.random.uniform(20, 25)
    agent.position_history = [agent.position]
    agent.velocity_history = [agent.velocity]
    agent.acceleration_history = [0.0]
    agent.time_history = [0.0]
    
    # Set epsilon to minimum for evaluation (minimal exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = agent.epsilon_min
    
    # Create renderer if needed
    renderer = None
    if render:
        renderer = TrafficSimulationRenderer(lane_length=lane_length)
    
    # Variables to track metrics
    dt = 0.1
    total_fuel = 0
    current_reward = 0
    
    # Simulation loop
    for step in range(steps):
        # Sort vehicles by position
        vehicles.sort(key=lambda v: v.position, reverse=True)
        
        # Update IDM vehicles
        for i, vehicle in enumerate(vehicles):
            lead_vehicle = vehicles[i-1] if i > 0 else None
            acceleration = vehicle.step(lead_vehicle, dt)
            total_fuel += vehicle.calculate_fuel_consumption(acceleration) * dt
        
        # Find the vehicle directly ahead of the RL agent
        lead_vehicle = None
        for vehicle in vehicles:
            if vehicle.position > agent.position:
                if lead_vehicle is None or vehicle.position < lead_vehicle.position:
                    lead_vehicle = vehicle
        
        # Get state and select action for RL agent
        state = agent.get_state(vehicles, lane_length)
        action_idx = agent.select_action(state)
        acceleration = agent.step(action_idx, lead_vehicle, dt)
        
        # Calculate fuel consumption for RL agent
        agent_fuel = agent.calculate_fuel_consumption(acceleration) * dt
        total_fuel += agent_fuel
        
        # Calculate reward components (for display only)
        velocity_reward = agent.velocity / 30.0
        smoothness_reward = -abs(acceleration) / 3.0
        fuel_reward = -agent.calculate_fuel_consumption(acceleration) / 5.0
        
        too_close = False
        if lead_vehicle:
            gap = lead_vehicle.position - agent.position - agent.length
            if gap < 5:
                too_close = True
        
        safety_penalty = -10.0 if too_close else 0.0
        current_reward = velocity_reward + smoothness_reward + fuel_reward + safety_penalty
        
        # Render if needed
        if render and step % render_interval == 0:
            renderer.render(vehicles, agent, step=step, 
                           reward=current_reward, total_fuel=total_fuel,
                           show_info=True)
            
            # Slow down rendering for visualization
            time.sleep(0.03)
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    # Close renderer if needed
    if renderer:
        renderer.close()
    
    # Generate time-space diagram
    plt.figure(figsize=(12, 8))
    
    # Plot IDM vehicles
    for i, vehicle in enumerate(vehicles):
        plt.plot(vehicle.time_history, vehicle.position_history, 'b-', alpha=0.5, linewidth=1)
    
    # Plot RL agent
    plt.plot(agent.time_history, agent.position_history, 'r-', linewidth=2, label='RL Agent')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Time-Space Diagram of Vehicle Trajectories')
    plt.legend()
    plt.grid(True)
    plt.savefig('time_space_diagram.png')
    
    # Plot velocity profiles
    plt.figure(figsize=(12, 8))
    
    # Plot IDM vehicles velocities
    for i, vehicle in enumerate(vehicles):
        plt.plot(vehicle.time_history, vehicle.velocity_history, 'b-', alpha=0.5, linewidth=1)
    
    # Plot RL agent velocity
    plt.plot(agent.time_history, agent.velocity_history, 'r-', linewidth=2, label='RL Agent')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Profiles of Vehicles')
    plt.legend()
    plt.grid(True)
    plt.savefig('velocity_profiles.png')
    
    # Plot rewards during training
    ##############################################################
    plt.figure(figsize=(12, 6))
    plt.plot(agent.rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.grid(True)
    plt.savefig('training_rewards.png')
    
    # Calculate and return metrics
    avg_speed = np.mean([np.mean(vehicle.velocity_history) for vehicle in vehicles])
    avg_rl_speed = np.mean(agent.velocity_history)
    
    print(f"Average speed of IDM vehicles: {avg_speed:.2f} m/s")
    print(f"Average speed of RL agent: {avg_rl_speed:.2f} m/s")
    print(f"Total fuel consumption: {total_fuel:.2f} ml")
    
    return vehicles, agent

def run_comparison_baseline(num_vehicles=10, lane_length=1000, steps=500):
    """Run a baseline simulation with only IDM vehicles for comparison"""
    vehicles = []
    
    # Create IDM vehicles with similar initial positions
    for i in range(num_vehicles + 1):  # +1 to match the total number of vehicles in the RL scenario
        pos = i * 15 + np.random.uniform(-2, 2)
        vel = np.random.uniform(20, 25)
        idm_params = {
            'v0': np.random.uniform(25, 30),
            'T': np.random.uniform(1.2, 1.8),
            'a': np.random.uniform(0.8, 1.2),
            'b': np.random.uniform(1.2, 1.8),
            'delta': 4.0,
            's0': np.random.uniform(1.5, 2.5),
            'position': pos, 
            'velocity': vel
        }
        vehicles.append(IDMVehicle(**idm_params))
    
    # Variables to track metrics
    dt = 0.1
    total_fuel = 0
    
    # Simulation loop
    for step in range(steps):
        # Sort vehicles by position
        vehicles.sort(key=lambda v: v.position, reverse=True)
        
        # Update IDM vehicles
        for i, vehicle in enumerate(vehicles):
            lead_vehicle = vehicles[i-1] if i > 0 else None
            acceleration = vehicle.step(lead_vehicle, dt)
            total_fuel += vehicle.calculate_fuel_consumption(acceleration) * dt
    
    # Generate time-space diagram for baseline
    plt.figure(figsize=(12, 8))
    
    # Plot IDM vehicles
    for i, vehicle in enumerate(vehicles):
        plt.plot(vehicle.time_history, vehicle.position_history, 'b-', alpha=0.5, linewidth=1)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Time-Space Diagram of Vehicle Trajectories (Baseline - IDM only)')
    plt.grid(True)
    plt.savefig('time_space_diagram_baseline.png')
    
    # Plot velocity profiles for baseline
    plt.figure(figsize=(12, 8))
    
    # Plot IDM vehicles velocities
    for i, vehicle in enumerate(vehicles):
        plt.plot(vehicle.time_history, vehicle.velocity_history, 'b-', alpha=0.5, linewidth=1)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Profiles of Vehicles (Baseline - IDM only)')
    plt.grid(True)
    plt.savefig('velocity_profiles_baseline.png')
    
    # Calculate and return metrics
    avg_speed = np.mean([np.mean(vehicle.velocity_history) for vehicle in vehicles])
    
    print(f"BASELINE - Average speed of IDM vehicles: {avg_speed:.2f} m/s")
    print(f"BASELINE - Total fuel consumption: {total_fuel:.2f} ml")
    
    return vehicles

def create_congestion_pattern(vehicles, num_steps=100, slow_down_factor=0.5):
    """Create an artificial congestion pattern by slowing down some vehicles"""
    # Choose a vehicle in the middle of the pack
    if len(vehicles) < 3:
        return
    
    target_idx = len(vehicles) // 2
    target_vehicle = vehicles[target_idx]
    
    # Temporarily reduce its desired speed to create a bottleneck
    original_v0 = target_vehicle.v0
    target_vehicle.v0 *= slow_down_factor
    
    # Temporarily increase its safe time headway to create more space
    original_T = target_vehicle.T
    target_vehicle.T *= 2.0
    
    # Simulate for a few steps to create congestion
    dt = 0.1
    for _ in range(num_steps):
        vehicles.sort(key=lambda v: v.position, reverse=True)
        for i, vehicle in enumerate(vehicles):
            lead_vehicle = vehicles[i-1] if i > 0 else None
            vehicle.step(lead_vehicle, dt)
    
    # Restore original parameters
    target_vehicle.v0 = original_v0
    target_vehicle.T = original_T

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Traffic Simulation with RL')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'baseline'],
                        help='Mode to run: train, eval, or baseline')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes for training')
    parser.add_argument('--steps', type=int, default=500, help='Number of steps per episode')
    parser.add_argument('--vehicles', type=int, default=10, help='Number of vehicles')
    parser.add_argument('--lane-length', type=int, default=1000, help='Length of the lane in meters')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--model-path', type=str, default='final_traffic_rl_model.pth',
                        help='Path to save/load the model')
    parser.add_argument('--congestion', action='store_true', help='Create artificial congestion')
    
    args = parser.parse_args()
    
    # Run based on mode
    if args.mode == 'train':
        print(f"Training RL agent for {args.episodes} episodes with {args.vehicles} vehicles...")
        trained_agent, final_vehicles = run_simulation_with_visualization(
            num_episodes=args.episodes,
            episode_steps=args.steps,
            num_vehicles=args.vehicles,
            lane_length=args.lane_length,
            render=args.render
        )
        
        print("\nTraining completed. Model saved to", args.model_path)
        print("Evaluating the trained agent...")
        
        # Evaluate the trained agent
        eval_vehicles, eval_agent = evaluate_traffic_flow_with_visualization(
            trained_agent,
            num_vehicles=args.vehicles,
            lane_length=args.lane_length,
            steps=args.steps,
            render=args.render
        )
        
    elif args.mode == 'eval':
        print("Evaluating pre-trained RL agent...")
        
        # Create RL agent and load the model
        state_dim = 8
        action_dim = 10
        agent = RLAgent(state_dim, action_dim)
        
        try:
            agent.load_model(args.model_path)
            print(f"Model loaded from {args.model_path}")
        except FileNotFoundError:
            print(f"Model file {args.model_path} not found. Please train a model first.")
            sys.exit(1)
        
        # Create vehicles
        vehicles = []
        for i in range(args.vehicles):
            pos = i * 15 + np.random.uniform(-2, 2)
            vel = np.random.uniform(20, 25)
            vehicles.append(IDMVehicle(position=pos, velocity=vel))
        
        # Create congestion if requested
        if args.congestion:
            print("Creating artificial congestion pattern...")
            create_congestion_pattern(vehicles)
        
        # Evaluate the agent
        eval_vehicles, eval_agent = evaluate_traffic_flow_with_visualization(
            agent,
            num_vehicles=args.vehicles,
            lane_length=args.lane_length,
            steps=args.steps,
            render=args.render
        )
        
    elif args.mode == 'baseline':
        print("Running baseline simulation with only IDM vehicles...")
        
        # Create vehicles
        vehicles = []
        for i in range(args.vehicles + 1):  # +1 to match the RL scenario's total vehicle count
            pos = i * 15 + np.random.uniform(-2, 2)
            vel = np.random.uniform(20, 25)
            vehicles.append(IDMVehicle(position=pos, velocity=vel))
        
        # Create congestion if requested
        if args.congestion:
            print("Creating artificial congestion pattern...")
            create_congestion_pattern(vehicles)
        
        # Create renderer if needed
        renderer = None
        if args.render:
            renderer = TrafficSimulationRenderer(lane_length=args.lane_length)
        
        # Variables to track metrics
        dt = 0.1
        total_fuel = 0
        
        # Simulation loop
        for step in range(args.steps):
            # Sort vehicles by position
            vehicles.sort(key=lambda v: v.position, reverse=True)
            
            # Update IDM vehicles
            for i, vehicle in enumerate(vehicles):
                lead_vehicle = vehicles[i-1] if i > 0 else None
                acceleration = vehicle.step(lead_vehicle, dt)
                total_fuel += vehicle.calculate_fuel_consumption(acceleration) * dt
            
            # Render if needed
            if args.render and step % 1 == 0:
                renderer.render(vehicles, None, step=step, total_fuel=total_fuel)
                time.sleep(0.03)
        
        # Close renderer if needed
        if renderer:
            renderer.close()
        
        # Run the comparison baseline
        baseline_vehicles = run_comparison_baseline(
            num_vehicles=args.vehicles,
            lane_length=args.lane_length,
            steps=args.steps
        )
    
    print("\nSimulation completed. Check the generated plots for visualizations.")