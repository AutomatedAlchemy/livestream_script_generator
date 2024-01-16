import json
import logging
import os
import random
import shutil
from logging.handlers import RotatingFileHandler

from flask import Flask, jsonify, request, send_file

from classes.SupportedScenes import \
    SupportedScenes  # Ensure this import path is correct

logging.basicConfig(level=logging.ERROR, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# Setup file handler
file_handler = RotatingFileHandler("./logs/logfile.log", maxBytes=1024 * 1024 * 100, backupCount=20)

app = Flask(__name__)
app.logger.addHandler(file_handler)

supported_scenes = None  # To store supported scenes data

@app.route("/setSupportedScenes", methods=["PUT"])
def set_supported_scenes():
    global supported_scenes
    data = request.json
    supported_scenes = SupportedScenes.from_json(json.dumps(data))
    return "Supported scenes set successfully!"

@app.route("/chooseEpisodePath", methods=["GET"])
def choose_episode_path():
    try:
        boosted_episodes = []  # Placeholder for boosted episodes
        unreleased_episodes = boosted_episodes + os.listdir("./shared/StreamingAssets/unreleased_episodes/")
        released_episodes = os.listdir("./shared/StreamingAssets/released_episodes/")

        if not unreleased_episodes:
            # Fallback to replaying old episodes
            episode_to_release = random.choice(released_episodes)
            shutil.copytree(f"./shared/StreamingAssets/released_episodes/{episode_to_release}",
                            f"./shared/StreamingAssets/unreleased_episodes/{episode_to_release}")
            shutil.rmtree(f"./shared/StreamingAssets/released_episodes/{episode_to_release}")

        # Selecting an episode from the top 3 of the list
        episode_to_release = random.choice(unreleased_episodes[:3])
        unity_episode_path = os.path.join("./shared/StreamingAssets/unreleased_episodes/", episode_to_release)
        app.logger.info(f"Selected episode for release: {unity_episode_path}")
        return jsonify({"episode_path": unity_episode_path})

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/getEpisode", methods=["GET"])
def get_episode():
    episode_path = request.args.get("path")
    try:
        with open(episode_path + "/script.json", "r") as file:
            episode_data = json.load(file)
        return jsonify(episode_data)
    except Exception as e:
        logger.exception(f"Error loading episode: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/getAudio", methods=["GET"])
def get_audio():
    episode_path = request.args.get("episodePath")
    character = request.args.get("character")
    action_index = request.args.get("actionIndex")
    try:
        audio_file_name = f"{action_index}_{character}.wav"
        audio_file_path = os.path.join(episode_path, audio_file_name)
        return send_file(audio_file_path, mimetype="audio/wav")
    except Exception as e:
        logger.exception(f"Error fetching audio file: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="localhost", port=5000)
