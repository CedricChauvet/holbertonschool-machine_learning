import numpy as np
import tensorflow as tf


class agent():
    def __init__(self, ALPHA, GAMMA=0.99, n_actions=2, layer1_size=16,
                 layer2_size=16, input_dims=4, filename='reinforce.h5'):
        self.gamma = GAMMA
        self.lr = ALPHA
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.policy_network = self.build_network()
        self.action_space = [i for i in range(n_actions)]
        self.model_file = filename

    def build_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.fc1_dims, activation='relu', 
                                   input_shape=(self.input_dims,)),
            tf.keras.layers.Dense(self.fc2_dims, activation='relu'),
            tf.keras.layers.Dense(self.n_actions, activation='softmax')
        ])
        return model

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.policy_network.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_space, p=probabilities)
        return action

    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def compute_loss(self, states, actions, advantages):
        # Compute policy network probabilities
        probs = self.policy_network(states)
        
        # Convert actions to one-hot encoding
        actions_one_hot = tf.one_hot(actions, self.n_actions)
        
        # Compute log probabilities of taken actions
        log_probs = tf.math.log(tf.reduce_sum(actions_one_hot * probs, axis=1))
        
        # Compute policy loss
        loss = -tf.reduce_mean(log_probs * advantages)
        
        return loss

    def learn(self):
        # Vérifier si la mémoire est vide
        if not self.state_memory:
            return 0.0

        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        # Compute discounted rewards (returns)
        G = np.zeros_like(reward_memory, dtype=float)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        # Normalize advantages
        advantages = (G - np.mean(G)) / (np.std(G) + 1e-8)

        # Use GradientTape for explicit gradient computation
        with tf.GradientTape() as tape:
            loss = self.compute_loss(
                tf.convert_to_tensor(state_memory, dtype=tf.float32),
                tf.convert_to_tensor(action_memory, dtype=tf.int32),
                tf.convert_to_tensor(advantages, dtype=tf.float32)
            )

        # Compute gradients and apply them
        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))

        # Clear memory
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        return loss.numpy()

    def save_model(self):
        self.policy_network.save(self.model_file)