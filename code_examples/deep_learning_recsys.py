
# Deep Learning Model using Keras for Ratings Prediction
import pandas as pd
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot
from keras.optimizers import Adam

# Load dataset (movieLens 100k as example)
# Dummy data simulation
num_users = 1000
num_items = 1700

ratings = pd.DataFrame({
    'user_id': [1, 2, 3, 1, 2],
    'item_id': [10, 20, 30, 40, 10],
    'rating': [5, 4, 3, 2, 4]
})

# Model parameters
embedding_size = 10

# Inputs
user_input = Input(shape=(1,), name='user_input')
item_input = Input(shape=(1,), name='item_input')

# Embedding layers
user_embedding = Embedding(num_users, embedding_size)(user_input)
item_embedding = Embedding(num_items, embedding_size)(item_input)

# Dot product of embeddings
user_vec = Flatten()(user_embedding)
item_vec = Flatten()(item_embedding)
dot_product = Dot(axes=1)([user_vec, item_vec])

# Build & compile model
model = Model(inputs=[user_input, item_input], outputs=dot_product)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train model (on small dummy data)
model.fit([ratings['user_id'], ratings['item_id']], ratings['rating'], epochs=10, verbose=1)
