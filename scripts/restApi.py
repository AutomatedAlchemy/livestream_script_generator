import os
import sys

# Add the root directory of your project to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import logging
import random
import shutil
from logging.handlers import RotatingFileHandler

from flask import Flask, jsonify, request, send_file

from classes.cls_poll import Poll
from classes.SupportedScenes import SupportedScenes


def list_full_paths(directory):
    """Returns the full paths of files in the given directory."""
    return [os.path.join(directory, file) for file in os.listdir(directory)]


logging.basicConfig(level=logging.ERROR, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

if not os.path.exists("./cache/logs/"):
    os.mkdir("./cache/logs/")
# Setup file handler
file_handler = RotatingFileHandler("./cache/logs/logfile.log", maxBytes=1024 * 1024 * 100, backupCount=20)

app = Flask(__name__)
app.logger.addHandler(file_handler)

supported_scenes: SupportedScenes  # To store supported scenes data


@app.route("/setSupportedScenes", methods=["PUT"])
def set_supported_scenes():
    global supported_scenes
    data = request.json
    supported_scenes = SupportedScenes.from_json(json.dumps(data))
    with open("./cache/shared/supported_scenes.json", "w") as file:
        file.write(supported_scenes.to_json())
    return "Supported scenes set successfully!"


@app.route("/chooseEpisodePath", methods=["GET"])
def choose_episode_path():
    try:
        prioritized_episodes_directory = "./cache/shared/StreamingAssets/prioritized_episodes/"
        unreleased_episodes_directory = "./cache/shared/StreamingAssets/unreleased_episodes/"
        released_episodes_directory = "./cache/shared/StreamingAssets/released_episodes/"

        prioritized_episodes = list_full_paths(prioritized_episodes_directory)  # Full paths for boosted episodes
        unreleased_episodes = prioritized_episodes + list_full_paths(unreleased_episodes_directory)  # Full paths for unreleased episodes
        released_episodes = list_full_paths(released_episodes_directory)  # Full paths for released episodes

        if not unreleased_episodes:
            # Fallback to replaying old episodes
            episode_to_play = random.choice(released_episodes)
            app.logger.info(f"Replaying episode: {episode_to_play}")
            print(f"Replaying episode: {episode_to_play}")
            released_path = episode_to_play
        else:
            # Selecting an episode from the top 3 of the list
            episode_to_play = random.choice(unreleased_episodes[:3])
            app.logger.info(f"Releasing episode: {episode_to_play}")
            print(f"Releasing episode: {episode_to_play}")
            released_path = os.path.join(released_episodes_directory, os.path.basename(episode_to_play))
            if os.path.exists(released_path):
                shutil.rmtree(released_path, True)
            shutil.move(episode_to_play, released_path)

        return jsonify({"episode_path": released_path})

    except Exception as e:
        app.logger.error(f"Error in choose_episode_path: {e}")
        return jsonify({"error": str(e)})


@app.route("/getEpisode", methods=["GET"])
def get_episode():
    episode_path = request.args.get("path")
    if not episode_path:
        raise Exception("path missing in get request")
    try:
        with open(episode_path + "/actions.json", "r") as file:
            episode_data = json.load(file)
        return jsonify(episode_data)
    except Exception as e:
        logger.exception(f"Error loading episode: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/getAudio", methods=["GET"])
def get_audio():
    episode_path = request.args.get("episodePath")
    if not episode_path:
        raise Exception("episode path missing in get request")
    character = request.args.get("character")
    action_index = request.args.get("actionIndex")
    try:
        audio_file_name = f"{action_index}_{character}.wav"
        audio_file_path = episode_path + "/" + audio_file_name
        return send_file("." + audio_file_path, mimetype="audio/wav")
    except Exception as e:
        logger.exception(f"Error fetching audio file: {e}")
        return jsonify({"error": str(e)}), 500

        
@app.route("/resetPoll", methods=["GET"])
def reset_poll():
    released_episodes_directory = "./cache/shared/StreamingAssets/released_episodes/"

    released_episodes = list_full_paths(released_episodes_directory)  # Full paths for released episodes

    next_episode_options = random.choices(released_episodes, k=3)
    poll: Poll  = Poll(next_episode_options)
    try:
        poll.to_file()
        return poll.to_json(), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/getPoll", methods=["GET"])
def get_poll():
    try:
        poll = Poll.from_file()
        return poll.to_json(), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="localhost", port=5000)
