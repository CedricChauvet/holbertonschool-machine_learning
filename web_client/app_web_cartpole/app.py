# app.py
from flask import Flask, render_template, Response
import tensorflow as tf
import numpy as np
import gymnasium as gym
import cv2
from io import BytesIO
import base64

app = Flask(__name__)

# Charger le modèle entraîné
model = tf.keras.models.load_model('cartpole_dqn_model')

# Créer l'environnement avec rgb_array pour capturer les images
env = gym.make('CartPole-v1', render_mode='rgb_array')
env.unwrapped.theta_threshold_radians = np.pi/2

def epsilon_greedy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)  # n_outputs = 2
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

def generate_frames():
    obs, _ = env.reset()
    while True:
        # Capture le rendu de l'environnement
        frame = env.render()
        
        # Convertir le frame numpy en image
        success, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_bytes = buffer.tobytes()
        
        # Prédire et exécuter l'action
        action = epsilon_greedy(obs, epsilon=0.01)
        obs, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            obs, _ = env.reset()
            
        # Envoyer l'image
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Changement ici : on écoute sur toutes les interfaces
    app.run(host='0.0.0.0', port=8080, debug=True)