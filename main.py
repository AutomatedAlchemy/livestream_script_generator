import argparse
import json
import logging
import os
import random
import re
import shutil
import threading
import time
from logging.handlers import RotatingFileHandler
from typing import Dict, List

import torch
from flask import Flask, jsonify, request
from TTS.api import TTS

from classes.Action import Action
from classes.Livestream import Livestream
from classes.SupportedScenes import SupportedScenes
from interface.cls_ollama_client import OllamaClient

if not os.path.exists("./logs"):
    os.mkdir("./logs")

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Run the script in different environments."
)
parser.add_argument(
    "-p", "--prod", action="store_true", help="Run in production environment"
)

args = parser.parse_args()

streaming_assets_path: str
if args.prod:
    print("Running in production mode")
    streaming_assets_path = ""
else:
    print("Running in development mode")
    streaming_assets_path = "C:/Users/Steffen/ai_livestream_URP/Assets/StreamingAssets/"

llm_logging: Dict[str, List[float]] = {}
current_llm_i: int = -1
current_episode_i: int = 0
episode_titles_to_choose_from = [
    "Logic in Computer Science",
    "Predicate Logic",
    "Natural Deduction",
    "Fitch Notation",
    "Coq Proof Assistant",
    "Group Theory",
    "Logical Paradoxes",
    "Practical Exercises",
    "Advanced Logic Topics (Optional)",
    "Advanced Predicate Logic",
    "Fitch-style Proof Techniques",
    "Coq Proof Assistant Mastery",
    "Group Theory Applications",
    "Logic Puzzles and Paradoxes",
    "The cult of done",
    "The cult of done manifesto",
    # "Boilerplate coding",
    # "Boilerplate programming",
    # "No Boilerplate programming as a paradigm",
    # "No Boilerplate",
    # "How to think about injectivity in math",
    # "How to think about surjectivity in math",
    # "How to think about bijectivity in math",
    # "mapping properties",
    # "injectivity",
    # "surjectivity",
    # "bijectivity",
    # "Math, Physics and other Metaphors",
    # "The Quantum Dance: Exploring Uncertainty",
    # "Minds and Machines: The Philosophy of AI",
    # "The Poetry of Physics: Unveiling the Cosmos",
    # "Alice's Algorithm: Adventures in Learning",
    # "Watts' Wisdom: The Nature of Consciousness",
    # "Feynman's Frontier: Discovering Nano Realms",
    # "Socratic Dialogues: AI and Ethics",
    # "Through the Looking Glass: Virtual Realities",
    # "Einstein's Echo: Relativity in Modern Times",
]
llms = [
    # "codellama:7b",
    # "deepseek-coder:6.7b",
    # "wizardcoder:7b",
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


def sanitize_filename(input_string):
    # Replace any non-alphanumeric character (excluding underscore) with an underscore
    sanitized_string = re.sub(r"\W+", "_", input_string)
    return sanitized_string


def synthesize_speech(
    text,
    output_folder_path: str,
    episode_identifier: str,
    voice_example_wav: str | None = None,
    model="tts_models/multilingual/multi-dataset/xtts_v2",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the TTS model
    tts = TTS(model, gpu=True, progress_bar=True).to(device)
    tts.tts_to_file(
        text=text,
        speaker_wav=voice_example_wav,
        language="en" if voice_example_wav else None,
        file_path=os.path.join(output_folder_path, episode_identifier),
        speed=1.2
    )


# initialize global variable
supported_scenes: SupportedScenes

# Initialize the Livestream object
livestream = Livestream(
    "Ai_Academia",
)


def generate_episodes():
    global llm_logging, current_llm_i, current_episode_i, llm
    while True:
        set_supported_scenes()
        
        while (
            len(os.listdir("./unreleased_episodes/")) > 10
            or not "supported_scenes" in globals()
        ):
            if not "supported_scenes" in globals():
                logger.exception(
                    "generate_episodes is waiting for supported_scenes to be initialized..."
                )
            time.sleep(10)
        try:
            
            # # Iterate through all titles and llms
            # if current_llm_i == len(llms) - 1:
            #     current_episode_i += 1
            #     current_llm_i = 0
            # else:
            #     current_llm_i += 1

            # if current_episode_i == len(episode_titles_to_choose_from) - 1:
            #     current_episode_i = 0
                
                
            # Iterate through all llms and titles
            if current_episode_i == len(episode_titles_to_choose_from) - 1:
                current_llm_i += 1
                current_episode_i = 0
            else:
                current_episode_i += 1

            if current_llm_i == len(llms) - 1:
                current_llm_i = 0
            
            #iterate first through all titles and then llms
            
            episode_title = episode_titles_to_choose_from[current_episode_i]
            llm = llms[current_llm_i]
            
            
            start_time = time.time()  # Start timer

            episode_identifier = sanitize_filename(f"{llm}_{episode_title}")

            WIP_path = f"./WIP_episode/{episode_identifier}"  # Clean working folder
            if os.path.exists("./WIP_episode/"):
                shutil.rmtree("./WIP_episode/")
                os.makedirs(WIP_path)

            episode = livestream.generate_episode(episode_title, supported_scenes, llm)
            script_generation_time = time.time() - start_time

            # write script to WIP
            with open(WIP_path + "/script.json", "w") as json_file:
                json_file.write(episode.to_json())
            # generate voices
            for i, action in enumerate(episode.script):
                if action.voice_line:
                    if action.character == "Richard Feynman":
                        synthesize_speech(
                            action.voice_line,
                            WIP_path,
                            f"{i}_{action.character}.wav",
                            "./voice_examples/FeynmanShort.wav",
                        )
                    elif action.character == "Alice":
                        synthesize_speech(
                            action.voice_line,
                            WIP_path,
                            f"{i}_{action.character}.wav",
                            model="tts_models/en/ljspeech/tacotron2-DDC",
                        )
                    elif action.character == "Alan Watts":
                        synthesize_speech(
                            action.voice_line,
                            WIP_path,
                            f"{i}_{action.character}.wav",
                            "./voice_examples/AlanWattsShort.wav",
                        )
                print(
                    f"\033[38;5;214mGenerating voices: {i+1}/{len(episode.script)}\033[0m"
                )
            
            # move Episode from WIP to ready
            episode_version = 0
            while os.path.exists(f"./unreleased_episodes/{episode_version}_{episode_identifier}") or os.path.exists(f"./released_episodes/{episode_version}_{episode_identifier}"):
                episode_version += 1
            shutil.copytree(WIP_path, f"./unreleased_episodes/{episode_version}_{episode_identifier}")
            print(WIP_path)
            print(f"./unreleased_episodes/{episode_version}_{episode_identifier}")
            shutil.rmtree(WIP_path)

            # logging llm info
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
                # app.logger.info(
                #     f"\033[38;5;255mAverage Time for {llm_name} to produce script: {average_time:.0f} seconds\033[0m"
                # )

        except Exception as e:
            logger.exception(f"An error occurred: {e}")
            time.sleep(1)



def set_supported_scenes():
    global supported_scenes

    # Assuming 'file_path' is a key in the JSON data of the POST request
    data = request.json
    file_path = data['file_path']

    # Reading the JSON data from the specified file
    with open(file_path, 'r') as file:
        file_data = json.load(file)

    print(file_data)
    supported_scenes = SupportedScenes.from_json(json.dumps(file_data))

    print("Supported scenes set successfully!")