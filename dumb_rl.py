import tetris
import keras
import numpy as np

model = keras.Sequential()
model.add(keras.layers.Dense(128, input_shape=(243,), activation="relu"))
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(7, activation="relu"))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

#naive data training
num_trials = 200000
min_score = 20
max_steps = 2000

game = tetris.Tetris(0, False)
training_x = []
training_y = []
scores = []
for trial in range(num_trials):
    x = []
    y = []
    game.reset(np.random.randint(2000000000))
    for _ in range(max_steps):
        inputs = []
        for _ in range(7):
            inputs += [np.random.randint(2)]
        x.append(game.get_state())
        y.append(inputs)

        fail = game.update_state(tuple(inputs))
        if fail:
            #print("fail")
            #print(game.get_score())
            break
    #print(game.get_score())
    scores.append(game.get_score())
    if game.get_score() > min_score:
        #print("hit")
        training_x += x
        training_y += y
    if trial % 1000 == 0:
        print("Trial: " + str(trial))
training_x, training_y = np.array(training_x), np.array(training_y)
print("Average Scores: " + str(np.average(scores)))
print("Max Score: " + str(np.amax(scores)))
print("Training Network")
#network code

model.fit(x=training_x, y=training_y, epochs=50)

model.save("my_model.h5")