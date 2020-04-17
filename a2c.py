import tetris
import numpy as np 
import tensorflow as tf 

games = 1000

explore_rate = 1 #represents how likely the agent is to take a random action
explore_decay = 0.999 #likelihood decays every game

actor_learning_rate = 0.003 #tunable parameters
critic_learning_rate = 0.005
discount_factor = 0.99

env = tetris.Tetris(False)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


actor = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(state_size,), activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(action_size, activation="softmax") #softmax outputs probability distribution
])
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)


critic = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(state_size,), activation="relu"),
    tf.keras.layers.Dense(1)
])
critic_optimizer = tf.keras.optimizers.Adam(learning_rate = critic_learning_rate)

def get_move(state):
    move_probabilities = actor(state)[0]
    if np.random.random_sample() > explore_rate: #probability of taking actual move
        return tf.math.argmax(move_probabilities).numpy()[0]
    else: 
        return np.random.randint(action_size)

actor_loss_function = tf.keras.losses.CategoricalCrossentropy()
def actor_loss(x, y):
    y_hat = actor(x)
    return actor_loss_function(y_pred=y_hat, y_true=y)

def actor_grad(x, y):
    with tf.GradientTape() as tape:
        loss_value = actor_loss(x, y)
    return tape.gradient(loss_value, actor.trainable_variables)

critic_loss_function = tf.keras.losses.MeanSquaredError()
def critic_loss(x, y):
    y_hat = critic(x)
    return critic_loss_function(y_pred=y_hat, y_true=y)

def critic_grad(x, y):
    with tf.GradientTape() as tape:
        loss_value = critic_loss(x, y)
    return tape.gradient(loss_value, critic.trainable_variables)

for game in range(games):
    done = False
    score = 0
    current_state = np.expand_dims(env.reset(), axis=0)

    while not done:
        current_move = get_move(current_state)
        next_state, reward, done, _ = env.step(current_move)
        next_state = np.expand_dims(next_state, axis=0)
        current_value = critic(current_state)[0]
        next_value = critic(next_state)[0]
        advantage = np.zeros(shape=(7,))
        if done:
            advantage[current_move] = -100 - current_value #reward for death is big negative number
            target_value = np.array([-100])
        else:
            td_error = reward + discount_factor * next_value - current_value
            advantage[current_move] = td_error #advantage actor critic - A2C
            target_value = np.array([td_error - current_value])
        actor_optimizer.apply_gradients(zip(actor_grad(current_state, advantage), actor.trainable_variables))
        critic_optimizer.apply_gradients(zip(critic_grad(current_state, advantage), critic.trainable_variables))
        score += reward
        state = next_state
        if reward > 1: #There must have been a line clear!
            print(reward)
        if done:
            print(str(game) + ": " + str(score))

    