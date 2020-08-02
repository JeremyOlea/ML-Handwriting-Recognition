import tensorflow as tf
import numpy as np
import pygame

TRAIN_DATA = True
train_x, train_y, test_x, test_y, model = None, None, None, None, None
CELL_SIZE = 30

drawing_grid = [[0 for __ in range(28)] for _ in range(28)]


def train():
    global test_x, test_y, model

    for data in train_x:
        for row in range(28):
            for col in range(28):
                if data[row][col] != 0:
                    data[row][col] = 1

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # range between 0 and 1
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=5)
    model.save('my_model.model')


def evaluate(model):
    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=1)
    print('Test accuracy:', test_acc)


def predict(model):
    # print(np.array([drawing_grid]).size)
    # print(np.array([test_x[0]]).size)
    for i in range(len(drawing_grid)):
        for j in range(len(drawing_grid[0])):
            test_x[0][i][j] = drawing_grid[i][j]
    # print(test_x[0])
    predictions = model.predict(np.array([test_x[0]]))
    print(np.argmax(predictions[0]))


def handle_mouse_click():
    pos = pygame.mouse.get_pos()
    col = pos[0] // CELL_SIZE
    row = pos[1] // CELL_SIZE
    # print(row, col)
    drawing_grid[row][col] = 1
    for i in range(row - 1, row + 1):
        if i < 28 and i > -1:
            for j in range(col - 1, col + 1):
                if j < 28 and j > -1:
                    drawing_grid[i][j] = 1


def run():
    pygame.display.set_caption("Number Drawer")
    pygame.init()
    screen = pygame.display.set_mode((840, 840))
    running = True
    closed = False
    for i in range(len(drawing_grid)):
        for j in range(len(drawing_grid[0])):
            drawing_grid[i][j] = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                closed = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = False
        if pygame.mouse.get_pressed()[0]:
            handle_mouse_click()
        spacing = CELL_SIZE
        for i in range(len(drawing_grid)):
            for j in range(len(drawing_grid[0])):
                if drawing_grid[i][j]:
                    pygame.draw.rect(screen, (0, 0, 0), [j * spacing, i * spacing, CELL_SIZE, CELL_SIZE])
                else:
                    pygame.draw.rect(screen, (255, 255, 255), [j * spacing, i * spacing, CELL_SIZE, CELL_SIZE])
        pygame.display.update()
        pygame.time.Clock().tick(1000)

    return closed


if __name__ == '__main__':
    dataset = tf.keras.datasets.mnist
    (train_x, train_y), (test_x, test_y) = dataset.load_data()

    if TRAIN_DATA:
        train()
    else:
        model = tf.keras.models.load_model('./my_model.model')

    for data in test_x:
        for row in range(28):
            for col in range(28):
                if data[row][col] != 0:
                    data[row][col] = 1

    while not run():
        predict(model)
    print("done")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
