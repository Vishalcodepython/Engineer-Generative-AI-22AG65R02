import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load and preprocess images
def load_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array / 255.0

content_img = load_image('path')
style_img = load_image('path/to')

# Model architecture
def build_model():
    model = Sequential()
    
    # Encoder
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    
    # Bottleneck
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    
    # Decoder
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    
    # Output layer
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
    
    return model

# Loss functions
def content_loss(content, generated):
    return tf.reduce_mean(tf.square(content - generated))

def gram_matrix(x):
    features = tf.reshape(x, (x.shape[0], -1, x.shape[3]))
    return tf.matmul(features, features, transpose_a=True) / tf.cast(x.shape[1]*x.shape[2]*x.shape[3], tf.float32)

def style_loss(style, generated):
    return tf.reduce_mean(tf.square(gram_matrix(style) - gram_matrix(generated)))

def total_loss(content, style, generated, alpha=1e-4, beta=1):
    return alpha * content_loss(content, generated) + beta * style_loss(style, generated)

# Training loop
num_epochs = 10
learning_rate = 1e-3

model = build_model()
optimizer = Adam(learning_rate)

for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        generated = model(content_img)
        loss = total_loss(content_img, style_img, generated)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.numpy()}')

# Save the trained model
model.save('style_transfer_model.h5')
