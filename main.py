import argparse
import json
import os
import shutil
import time
from typing import Dict, List

import torch
from flask import Flask, jsonify, request
from TTS.api import TTS

from classes.Action import Action
from classes.Livestream import Livestream
from classes.SupportedScenes import SupportedScenes
from interface.cls_ollama_client import OllamaClient

parser = argparse.ArgumentParser(description='Run the script in different environments.')
parser.add_argument('-p', '--prod', action='store_true', help='Run in production environment')
args = parser.parse_args()
streaming_assets_path:str
if args.prod:
    print("Running in production mode")
    streaming_assets_path = "C:/Users/Steffen/ai_livestream_URP/Assets/StreamingAssets/"
else:
    print("Running in development mode")
    streaming_assets_path = "C:/Users/Steffen/ai_livestream_URP/Assets/StreamingAssets/"

llm_logging: Dict[str, List[float]] = {}
current_llm_i: int = -1
current_episode_i: int = 0
episode_titles = [
    "Math, Physics and other Metaphors",
    "The Quantum Dance: Exploring Uncertainty",
    "Minds and Machines: The Philosophy of AI",
    "The Poetry of Physics: Unveiling the Cosmos",
    "Alice's Algorithm: Adventures in Learning",
    "Watts' Wisdom: The Nature of Consciousness",
    "Feynman's Frontier: Discovering Nano Realms",
    "Socratic Dialogues: AI and Ethics",
    "Through the Looking Glass: Virtual Realities",
    "Einstein's Echo: Relativity in Modern Times",
]
llms = [
    "zephyr",
    "starling-lm",
    "neural-chat",
    "llama2-uncensored",
    "openhermes",
    "orca2",
]

for llm_to_download in llms:
    session = OllamaClient()
    session._download_model(llm_to_download)

llm = llms[0]


def simplify_json(json: str):
    while "\n" in json or "\t" in json or "  " in json:
        json = json.replace("\n", "").replace("\t", "").replace("  ", " ")
    return json


def synthesize_speech(
    text,
    output_folder_path: str,
    episode_identifier: str,
    speaker_wav: str | None = None,
    model="tts_models/multilingual/multi-dataset/xtts_v2",
    export_to_unity: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the TTS model
    tts = TTS(model, gpu=True, progress_bar=True).to(device)
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language="en" if speaker_wav else None,
        file_path=os.path.join(output_folder_path,episode_identifier),
    )
    if export_to_unity:
        unity_episode_path = f"{streaming_assets_path}{episode_identifier}"
        if not os.path.exists(unity_episode_path):
            os.makedirs(unity_episode_path)
        try:
            shutil.copy(output_folder_path, f"{unity_episode_path}/{output_folder_path}")
        except Exception as e:
            print(f"Failed to copy file: {e}")


# initialize global variable
supported_scenes: SupportedScenes

app = Flask(__name__)

# Initialize the Livestream object
livestream = Livestream(
    "Ai_Academia",
)


@app.route("/get_episode_path", methods=["GET"])
def get_episode():
    global llm_logging, current_llm_i, current_episode_i, llm

    if current_llm_i == len(llms) - 1:
        current_episode_i += 1
        current_llm_i = 0
    else:
        current_llm_i += 1

    if current_episode_i == len(episode_titles) - 1:
        current_episode_i = -1
    episode_title = episode_titles[current_episode_i]
    llm = llms[current_llm_i]
    start_time = time.time()  # Start timer

    episode_identifier = f"{llm}_{episode_title}"
    
    output_folder_path = f"./outputs/{episode_identifier}"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    try:
        episode = livestream.generate_episode(episode_title, supported_scenes, llm)
        script_generation_time = time.time() - start_time

        for i, action in enumerate(episode.script):
            if action.character == "Richard Feynman":
                synthesize_speech(
                    action.voice_line,
                    output_folder_path,
                    f"{i}_{action.character}.wav",
                    "./FeynmanShort.wav",
                )
            elif action.character == "Alice":
                synthesize_speech(
                    action.voice_line,
                    output_folder_path,
                    f"{i}_{action.character}.wav",
                    model="tts_models/en/ljspeech/tacotron2-DDC",
                )
            elif action.character == "Alan Watts":
                synthesize_speech(
                    action.voice_line,
                    output_folder_path,
                    f"{i}_{action.character}.wav",
                    "./AlanWattsShort.wav",
                )
            print(
                f"\033[38;5;214mGenerating voices: {i+1}/{len(episode.script)}\033[0m"
            )

        with open(os.path.join(output_folder_path, "script.json"), "w") as json_file:
            json_file.write(episode.to_json())
        unity_episode_path = f"{streaming_assets_path}{episode_identifier}"

        # logging
        if script_generation_time > 3:
            if llm in llm_logging:
                llm_logging[llm].append(script_generation_time)
            else:
                llm_logging[llm] = [script_generation_time]

        # logging: Script speed evaluation printing
        for llm_name in llm_logging.keys():
            average_time = sum(llm_logging[llm_name]) / len(llm_logging[llm_name])
            print(
                f"\033[38;5;255mAverage Time for {llm_name} to produce script: {average_time:.0f} seconds\033[0m"
            )
            
        episode_path_info = json.dumps({"episode_path": unity_episode_path})
        app.logger.info(f"get_episode_path: {episode_path_info}")

        return jsonify({"episode_path": unity_episode_path})
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")

        return f"A python-side error ocurred: {e}"


@app.route("/set_supported_scenes", methods=["POST"])
def set_supported_scenes():
    global supported_scenes
    data = request.json
    print(data)
    supported_scenes = SupportedScenes.from_json(json.dumps(data))

    return jsonify({"message": "Supported scenes received successfully"})


if __name__ == "__main__":
    app.run(debug=False, host="localhost", port=5000)
