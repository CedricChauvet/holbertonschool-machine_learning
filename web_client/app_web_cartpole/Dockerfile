# Dockerfile
FROM python:3.9-slim

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Création du répertoire de travail
WORKDIR /app

# Copie des fichiers requis
COPY requirements.txt .
COPY app.py .
COPY templates/ templates/
COPY cartpole_dqn_model/ cartpole_dqn_model/

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposition du port
EXPOSE 8080

# Commande de démarrage
CMD ["python", "app.py"]