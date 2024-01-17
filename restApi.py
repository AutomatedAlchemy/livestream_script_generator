import json
import logging
import os
import random
import shutil
from logging.handlers import RotatingFileHandler

from flask import Flask, jsonify, request, send_file

from classes.SupportedScenes import \
    SupportedScenes  # Ensure this import path is correct


def list_full_paths(directory):
    """ Returns the full paths of files in the given directory. """
    return [os.path.join(directory, file) for file in os.listdir(directory)]


logging.basicConfig(level=logging.ERROR, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

if (not os.path.exists("./logs/")):
    os.mkdir("./logs/")
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
    json.dump(supported_scenes, "./shared/supported_scenes.json")
    return "Supported scenes set successfully!"


@app.route("/chooseEpisodePath", methods=["GET"])
def choose_episode_path():
    try:
        boosted_episodes_directory = "./shared/StreamingAssets/prioritized_episodes/"
        unreleased_episodes_directory = "./shared/StreamingAssets/unreleased_episodes/"
        released_episodes_directory = "./shared/StreamingAssets/released_episodes/"

        boosted_episodes = list_full_paths(boosted_episodes_directory)  # Full paths for boosted episodes
        unreleased_episodes = boosted_episodes + list_full_paths(unreleased_episodes_directory)  # Full paths for unreleased episodes
        released_episodes = list_full_paths(released_episodes_directory)  # Full paths for released episodes

        if not unreleased_episodes:
            # Fallback to replaying old episodes
            episode_to_release = random.choice(released_episodes)
        else:
            # Selecting an episode from the top 3 of the list
            episode_to_release = random.choice(unreleased_episodes[:3])
        released_path = os.path.join(released_episodes_directory, os.path.basename(episode_to_release))
        if (not os.path.exists(released_path)):
            shutil.copytree(episode_to_release, released_path)
        if (not "released_episodes" in episode_to_release):
            shutil.rmtree(episode_to_release)
        app.logger.info(f"Releasing episode: {os.path.basename(released_path)}")
        return jsonify({"episode_path": released_path})
    
    except Exception as e:
        app.logger.error(f"Error in choose_episode_path: {e}")
        return jsonify({"error": str(e)})


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
