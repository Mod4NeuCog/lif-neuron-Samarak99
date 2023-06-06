import random


# 1 - This class create the environment of the game
class GameShowEnv:
    def __init__(self):
        self.state = 1  # Initial state (Q1)
        self.reward = 0
        self.done = False

    def reset(self):
        self.state = 1  # Reset to initial state (Q1)
        self.reward = 0
        self.done = False

    def step(self, action):
        if self.state == 1:  # Q1
            if random.random() < 0.9:  # 90% chance of correct answer
                self.state = 2  # Move to Q2
                self.reward = 100  # Win $100
            else:
                self.reward = 0  # Lose the game
                self.done = True

        elif self.state == 2:  # Q2
            if random.random() < 0.75:  # 75% chance of correct answer
                self.state = 3  # Move to Q3
                self.reward = 1000  # Win $1000
            else:
                self.reward = 100  # Go back to Q1 with $100
                self.state = 1
                self.done = True

        elif self.state == 3:  # Q3
            if random.random() < 0.5:  # 50% chance of correct answer
                self.state = 4  # Move to Q4
                self.reward = 10000  # Win $10000
            else:
                self.reward = 1100  # Go back to Q1 with $1100
                self.state = 1
                self.done = True

        elif self.state == 4:  # Q4
            if random.random() < 0.1:  # 10% chance of correct answer
                self.state = 50000  # Win $50000 and end the game
                self.reward = 50000
                self.done = True
            else:
                self.reward = 11100  # Go back to Q1 with $11100
                self.state = 1
                self.done = True

        return self.state, self.reward, self.done

## when the user loses in any state, they will go back to Q1 with the corresponding reward,
# and the game will be marked as done. This allows the agent to learn from both successful and unsuccessful attempts.
# BOOSTING REINFORCEMENT LEARNING


# here we are building the agent that gonna play the game, and implement q-learning
class Agent:
    def __init__(self, alpha, gamma):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = {}

    def get_q_value(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        return self.q_table[(state, action)]

    def update_q_value(self, state, action, new_q_value):
        self.q_table[(state, action)] = new_q_value

    def choose_action(self, state):
        if random.random() < 0.9:  # 90% chance of selecting the best action
            max_q_value = float('-inf')
            best_action = None

            for action in [0, 1]:
                q_value = self.get_q_value(state, action)
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action

            return best_action
        else:  # 10% chance of selecting a random action
            return random.choice([0, 1])

    # For episode, it corresponds to a single attempt of the game by the agent - so we can see how much he takes to learn.
    def q_learning(self, env, episodes):
        for episode in range(episodes):
            env.reset()
            state = env.state

            while not env.done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)

                q_value = self.get_q_value(state, action)
                max_next_q_value = max(self.get_q_value(next_state, a) for a in [0, 1])

                new_q_value = q_value + self.alpha * (reward + self.gamma * max_next_q_value - q_value)
                self.update_q_value(state, action, new_q_value)

                state = next_state

            if (episode + 1) % 100 == 0:
                print("Episode:", episode + 1)
                print("Q-table:", self.q_table)


# Create game show environment
env = GameShowEnv()

# Create agent and perform Q-learning
# alpha is the learning rate
# gamma : disocunt factor : show the importance given to future rewards compared to immediate rewards.
agent = Agent(alpha=0.5, gamma=0.9)
agent.q_learning(env, episodes=1000)