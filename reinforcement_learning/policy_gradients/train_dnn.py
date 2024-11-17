import numpy as np
import tensorflow as tf

class PolicyNetwork:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(4,), trainable=True),
            tf.keras.layers.Dense(2, activation='softmax', trainable=True)
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    def __call__(self, state):
        return self.model(state)
    
def policy_gradient(state, policy_network):
    """
    Calculate policy gradient using current policy network
    """
    state = np.array([state])  # Add batch dimension
    policy_network.trainable = True

    with tf.GradientTape() as tape:
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        policy_probs = policy_network.model(state_tensor)
        numpy_policy =policy_probs.numpy()[0]
        # Sample action from policy
        if np.random.random() > numpy_policy[0]:
            action = 1
        else:
            action = 0
    
        # Calculate log probability of the taken action
        # action_mask = tf.one_hot(action, depth=2)
        # log_prob = tf.math.log(tf.reduce_sum(policy_probs * action_mask))
    
        # Get gradients of log probability
        gradients = tape.gradient(numpy_policy, policy_network.model.trainable_variables)
    # Convertir les gradients en numpy
        gradient_numpy = [g.numpy() for g in gradients if g is not None]
        print("gradient_numpy: ", gradient_numpy[0].shape)
    return int(action), gradient_numpy[0]


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    θ(t+1) = θ(t) + α ∑(γ^t ∇θ log π(at|st; θ) Rt)
    """
    scores = []
    policy_network = PolicyNetwork()
    n_states, n_actions = env.observation_space.shape[0], env.action_space.n
    
    weights = np.random.rand(n_states, 64, n_actions)
    # grad = np.zeros(weights.shape)


    for i in range(nb_episodes):
        state, _ = env.reset()
        done = False
        # print("state: ", state)
        rewards = []
        gradients = []
        while not done:
            action, grad = policy_gradient(state, policy_network)
            print("action", action)
            print("grad: ", grad.shape)
            
            next_state, reward, done, _, _ = env.step(action)
            gradients.append(grad)
            rewards.append(reward)
            
            state = next_state
            
            weights += alpha * sum([grad * (gamma ** t) * reward for t, (grad, reward) in enumerate(zip(gradients, rewards))])
            
            print("weights: ", weights.shape)
            policy_network.optimizer.apply_gradients(zip(weights, policy_network.model.trainable_variables))
        # weights += alpha * sum([grad * (gamma ** t)   for t, (grad) in enumerate(grads)])
        
        scores.append(sum(rewards))
        print("EP: " + str(i) + " Score: " + str(sum(rewards)))
    return scores


