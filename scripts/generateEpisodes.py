import os
import sys

# Add the root directory of your project to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import builtins
import json
import os
import random
import re
import shutil
import time
import traceback
from random import shuffle
from typing import Dict, List

import torch
from TTS.api import TTS

from classes.Episode import Episode
from classes.Livestream import Livestream
from classes.SupportedScenes import SupportedScenes
from interface.cls_few_shot_factory import FewShotProvider
from interface.cls_livestream_message import LivestreamMessage
from interface.cls_ollama_client import OllamaClient

# Save the original print function
original_print = builtins.print


# Define a new print function that always flushes
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return original_print(*args, **kwargs)


# Override the built-in print with your custom print
builtins.print = print

if not os.path.exists("./logs"):
    os.mkdir("./logs")

parser = argparse.ArgumentParser(description="Run the script in different environments.")
parser.add_argument("-p", "--prod", action="store_true", help="Run in production environment")

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
    "RISC-V"
    "RISC Architecture",
    "Assembler Programming",
    # "The Role of the Operating System",

    
    # "Generalization of Numbers to Matrices",
    # "Fractals",
    # "Cellular Automata",
    # "Mathematical Spirals",
    # "Random Walks",
    # "Large Language Models",
    # "GPT-4",
    # "GPT-5",
    # "Neuroplasticity",
    # "Evolutionary Psychology",
    # "Ilya Sutskever",
    # "Quantum Entanglement and Nonlocality",
    # "The Nature of Consciousness",
    # "Chaos Theory and the Butterfly Effect",
    # "The Fibonacci Sequence in Nature",
    # "Entropy and the Arrow of Time",
    # "The Multiverse Theory",
    # "The Evolution of Gender Roles Through History",
    # "The Ethics of Artificial Intelligence",
    # "Climate Change: A Scientific and Philosophical Inquiry",
    # "Privacy in the Age of Big Data",
    # "The Future of Work in an AI-Driven World",
    # "Universal Basic Income: Economic Solution or Utopian Dream?",
    # "The Intersection of Art and Science",
    # "Ethical Considerations in Genetic Engineering",
    # "Exploring the Unconscious with Carl Jung",
    # "The Legacy of Sigmund Freud on Modern Psychology",
    # "The Rise of Nvidia and the GPU Revolution",
    # "Microsoft's Journey from DOS to the Cloud",
    # "OpenAI and the Quest for Responsible AI",
    # "Browser Wars: Firefox vs Chrome",
    # "The Science and Philosophy of Nutrition",
    # "Cryptography: From Ancient Codes to Quantum Hacking",
    # "Social Media's Influence on Public Opinion",
    # "Cultural Imperialism and Globalization",
    # "The Psychology of Power and Leadership"
]

# https://huggingface.co/TheBloke/NeuralBeagle14-7B-GGUF
# https://huggingface.co/senseable/WestLake-7B-v2
llms = [
    # "dolphin-mixtral",
    # "samantha-mistral",
    # "zephyr",
    # "starling-lm",
    # "neural-chat",
    # "llama2-uncensored",
    # "openhermes",
    # "orca2",
    "phi3",
]
shuffle(episode_titles_to_choose_from)
shuffle(llms)

session = OllamaClient()
for llm_to_download in ["llava:v1.6", "codellama", "wizardcoder"] + llms:
    session._download_model(llm_to_download)
session._restart_container()

llm = llms[0]


def simplify_json(json: str) -> str:
    while "\n" in json or "\t" in json or "  " in json:
        json = json.replace("\n", "").replace("\t", "").replace("  ", " ")
    return json


def sanitize_filename(input_string: str) -> str:
    # Replace any non-alphanumeric character (excluding underscore) with an underscore
    sanitized_string = re.sub(r"\W+", "_", input_string)
    return sanitized_string


def synthesize_speech(
    text: str,
    output_folder_path: str,
    episode_identifier: str,
    voice_example_wav: str | None = None,
    model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the TTS model
    tts = TTS(model, gpu=True, progress_bar=True).to(device)
    tts.tts_to_file(text=text, speaker_wav=voice_example_wav, language="en" if voice_example_wav else None, file_path=os.path.join(output_folder_path, episode_identifier), speed=1.2)


# initialize global variable
supported_scenes: SupportedScenes

# Initialize the Livestream object
livestream = Livestream(
    "Ai_Academia",
)


def generate_episode(episode_title: str) -> None:
    try:
        llm = random.choice(llms)
        episode_title = random.choice(episode_titles_to_choose_from)

        start_time = time.time()  # Start timer

        episode_identifier = sanitize_filename(f"{llm}_{episode_title}")

        WIP_path = f"./cache/WIP_episode/{episode_identifier}"  # Clean working folder
        if os.path.exists("./cache/WIP_episode/"):
            shutil.rmtree("./cache/WIP_episode/")
            os.makedirs(WIP_path)

        episode = livestream.generate_episode(episode_title, supported_scenes, llm)

        actions_generation_time = time.time() - start_time

        # write actions to WIP
        with open(WIP_path + "/actions.json", "w") as json_file:
            json_file.write(episode.to_json())
        # generate voices
        for i, action in enumerate(episode.actions):
            if action.voice_line:
                if "Feynman" in action.character or "Richard" in action.character:
                    synthesize_speech(
                        action.voice_line,
                        WIP_path,
                        f"{i}_{action.character}.wav",
                        "./voice_examples/FeynmanShort.wav",
                    )
                elif "Alice" in action.character:
                    synthesize_speech(
                        action.voice_line,
                        WIP_path,
                        f"{i}_{action.character}.wav",
                        model="tts_models/en/ljspeech/tacotron2-DDC",
                    )
                elif "Watts" in action.character or "Alan" in action.character:
                    synthesize_speech(
                        action.voice_line,
                        WIP_path,
                        f"{i}_{action.character}.wav",
                        "./voice_examples/AlanWattsShort.wav",
                    )
            print(f"\033[38;5;214mGenerating voices: {i+1}/{len(episode.actions)}\033[0m")

        # move Episode from WIP to ready
        episode_version = 0
        while (
            os.path.exists(f"./cache/shared/StreamingAssets/unreleased_episodes/{episode_version}_{episode_identifier}")
            or os.path.exists(f"./cache/shared/StreamingAssets/prioritized_episodes/{episode_version}_{episode_identifier}")
            or os.path.exists(f"./cache/shared/StreamingAssets/released_episodes/{episode_version}_{episode_identifier}")
        ):
            episode_version += 1
        generated_episode_folder = f"./cache/shared/StreamingAssets/prioritized_episodes/{episode_version}_{episode_identifier}"
        shutil.copytree(WIP_path, generated_episode_folder)
        print(WIP_path)
        print(generated_episode_folder)
        shutil.rmtree(WIP_path)

        # logging llm info
        if actions_generation_time > 3:
            if llm in llm_logging:
                llm_logging[llm].append(actions_generation_time)
            else:
                llm_logging[llm] = [actions_generation_time]

        # logging: Action speed evaluation printing
        for llm_name in llm_logging.keys():
            average_time = sum(llm_logging[llm_name]) / len(llm_logging[llm_name])
            print(f"\033[38;5;255mAverage Time for {llm_name} to produce actions: {average_time:.0f} seconds\033[0m")

    except Exception as e:
        print("\033[91mAn error occurred:\033[0m", e)

        # Print file name and line number
        tb = traceback.extract_tb(e.__traceback__)
        filename, line, func, text = tb[-1]
        print(f"\033[93mFile: {filename}, Line: {line}, In: {func}\033[0m")

        # Separately print the full call stack
        print("\033[94mCall Stack:\033[0m")
        print("".join(traceback.format_tb(e.__traceback__)))
        raise (e)
        time.sleep(1)


def set_supported_scenes() -> None:
    global supported_scenes
    # Reading the JSON data from the specified file
    with open("./cache/shared/supported_scenes.json", "r") as file:
        file_data = file.read()

    print(file_data)
    supported_scenes = SupportedScenes.from_json(file_data)
    print("Supported scenes set successfully!")


def validate_generated_episodes() -> None:
    episode_paths: list[str] = []
    for dir_path in os.listdir("./cache/shared/StreamingAssets/"):
        episode_paths += [f"./cache/shared/StreamingAssets/{dir_path}/{episode_file}" for episode_file in os.listdir("./cache/shared/StreamingAssets/" + dir_path)]
    for episode_path in episode_paths:
        json_path = episode_path + "/actions.json"
        try:
            with open(json_path, "r") as file:
                json_content = json.load(file)
        except Exception as e:
            print(f"\033[91mDELETING FAULTY EPISODE\tREASON: {e}\n{episode_path}\033[0m")
            shutil.rmtree(episode_path)
            continue

        try:
            episode = Episode.from_json(json.dumps(json_content), llms[0])
            if (len(episode.actions) < 5):
                shutil.rmtree(episode_path)
        except Exception as e:
            print(f"Error processing episode: {e}")
            continue


# def chat_to_topics(newMessages_json_path: str = "./newMessages.json"):
#     newMessages: list[LivestreamMessage] = []
#     with open(newMessages_json_path) as file:
#         newMessages = json.load(file)
#     topics: list[str] = FewShotProvider.few_shot_LivestreamMessagesToTopics(newMessages, llm)

validate_generated_episodes()

# chat_based_episode_titles: list[str] = chat_to_topics()
# generate_episodes(chat_based_episode_titles)


while True:
    set_supported_scenes()
    episode_title:str = random.choice(episode_titles_to_choose_from)
    generate_episode(episode_title)
