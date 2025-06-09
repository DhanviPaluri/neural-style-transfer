import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import time

# Load and preprocess image
def load_and_process_image(path, max_dim=512):
    img = load_img(path)
    img = img_to_array(img)
    img = tf.image.resize(img, (max_dim, max_dim))
    img = vgg19.preprocess_input(img)
    return tf.expand_dims(img, axis=0)

# Deprocess image
def deprocess_image(processed_img):
    x = processed_img.copy()
    x = x.reshape((x.shape[1], x.shape[2], 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # BGR to RGB
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Load content and style images
content_path = "content.jpg"
style_path = "style.jpg"

content_image = load_and_process_image(content_path)
style_image = load_and_process_image(style_path)

# Use VGG19
vgg = vgg19.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Get style and content layers
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                'block4_conv1', 'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Model to extract features
def get_model():
    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# Gram matrix for style
def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

def get_features(model, content, style):
    inputs = tf.concat([style, content], axis=0)
    outputs = model(inputs)
    style_outputs, content_outputs = outputs[:num_style_layers], outputs[num_style_layers:]

    style_features = [gram_matrix(style_layer) for style_layer in style_outputs]
    content_features = [content_layer for content_layer in content_outputs]

    return style_features, content_features

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    outputs = model(init_image)
    style_output_features = outputs[:num_style_layers]
    content_output_features = outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += tf.reduce_mean((gram_matrix(comb_style) - target_style) ** 2)

    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += tf.reduce_mean((comb_content - target_content) ** 2)

    style_score *= style_weight / num_style_layers
    content_score *= content_weight / num_content_layers

    loss = style_score + content_score
    return loss

@tf.function()
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        loss = compute_loss(**cfg)
    grads = tape.gradient(loss, cfg['init_image'])
    return grads, loss

def run_style_transfer(content_image, style_image, iterations=1000, style_weight=1e-2, content_weight=1e4):
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    style_features, content_features = get_features(model, content_image, style_image)
    init_image = tf.Variable(content_image, dtype=tf.float32)
    opt = tf.optimizers.Adam(learning_rate=5.0)

    best_loss, best_img = float('inf'), None

    cfg = {
        'model': model,
        'loss_weights': (style_weight, content_weight),
        'init_image': init_image,
        'gram_style_features': style_features,
        'content_features': content_features
    }

    print("Starting style transfer...")

    for i in range(iterations):
        grads, loss = compute_grads(cfg)
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, -103.939, 255.0 - 103.939)
        init_image.assign(clipped)

        if loss < best_loss:
            best_loss = loss
            best_img = deprocess_image(init_image.numpy())

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")

    return best_img

stylized_image = run_style_transfer(content_image, style_image, iterations=500)

output_image = PIL.Image.fromarray(stylized_image)
output_image.save("output.jpg")
plt.imshow(output_image)
plt.title("Stylized Image")
plt.axis('off')
plt.show()
