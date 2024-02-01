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
from line_profiler import LineProfiler, profile
from TTS.api import TTS

from classes.Episode import Episode
from classes.Livestream import Livestream
from classes.SupportedScenes import SupportedScenes
from interface.cls_ollama_client import OllamaClient

# Save the original print function
original_print = builtins.print

# Define a new print function that always flushes
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    return original_print(*args, **kwargs)

# Override the built-in print with your custom print
builtins.print = print

if not os.path.exists("./logs"):
    os.mkdir("./logs")

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
    "Linear Mappings", 
    "Inverses of Matrices", 
    "Generalization of Numbers to Matrices", 
    "Transposing and Inverting Matrices, Determinant of the Transposed Matrix", 
    "Powers and Inverses of Matrices", 
    "Solving Linear Systems", 
    "Formula for Inverse of a 2x2 Matrix", 
    "Applications of Matrices", 
    "Sum Formula for Geometric Sequences Applied to Matrices"
    "RasperryPi Pico",
    "Mandelbrot Set",
    "Julia Sets",
    "Fractals",
    "Mathematical Functions",
    "Chaos Theory and Attractors",
    "Cellular Automata",
    "Fourier Series Visualization",
    "Mathematical Spirals",
    "Waveform Patterns",
    "Pendulum Motion",
    "Lissajous Curves",
    "Random Walks",
    "Turing Patterns",
    "The Cult of Done",
    "The Cult of Done Manifesto",
    "Large Language Models",
    "ChatGpt",
    "GPT-5",
    "OpenAI",
    "The Future After the Singularity of AI",
    "DNA Sequencing Technologies",
    "Stem Cell Research",
    "Neuroplasticity",
    "Evolutionary Psychology",
    "Social Network Analysis",
    "Cultural Anthropology",
    "Behavioral Economics",
    "Game Theory in Economics",
    "Virtual Reality Gaming",
    "Esports and Competitive Gaming",
    "History of Video Games",
    "Artificial Intelligence in Gaming",
    "Psychological Impact of Social Media",
    "Renewable Energy Sources",
    "Climate Change and its Effects",
    "Space Exploration and Colonization",
    "Quantum Computing",
    "Blockchain and Cryptocurrency",
    "Cybersecurity Trends and Challenges",
    "Home Gardening Tips",
    "Easy Healthy Recipes",
    "Budget Travel Destinations",
    "DIY Home Decor Ideas",
    "Effective Workout Routines for Beginners",
    "Mindfulness and Meditation Practices",
    "Time Management Strategies",
    "Book Recommendations",
    "Eco-Friendly Living Practices",
    "Quick and Easy Meal Prepping",
    "Home Office Organization",
    "Stress Relief Techniques",
    "Indoor Plant Care",
    "Upcycling Projects",
    "Coffee Brewing Techniques",
    "Crafting and Scrapbooking Ideas",
    "Board Games and Family Entertainment",
    "Learning a New Language",
    "Urban Gardening",
]
# https://huggingface.co/TheBloke/NeuralBeagle14-7B-GGUF
# https://huggingface.co/senseable/WestLake-7B-v2
llms = [
    # "dolphin-mixtral",
    "samantha-mistral",
    "zephyr",
    "starling-lm",
    "neural-chat",
    "llama2-uncensored",
    "openhermes",
    "orca2",
]
shuffle(episode_titles_to_choose_from)
shuffle(llms)

session = OllamaClient()
for llm_to_download in ["bakllava:7b-v1-q4_K_M", "phi", "codellama", "wizardcoder"] + llms:
    session._download_model(llm_to_download)
session._restart_container()

llm = llms[0]


def simplify_json(json: str):
    while "\n" in json or "\t" in json or "  " in json:
        json = json.replace("\n", "").replace("\t", "").replace("  ", " ")
    return json


def sanitize_filename(input_string):
    # Replace any non-alphanumeric character (excluding underscore) with an underscore
    sanitized_string = re.sub(r"\W+", "_", input_string)
    return sanitized_string

@profile
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

@profile
def generate_episodes():
    global llm_logging, current_llm_i, current_episode_i, llm
    while True:
        set_supported_scenes()
        try:
            # Iterate through all llms and titles
            if current_episode_i == len(episode_titles_to_choose_from) - 1:
                current_llm_i += 1
                current_episode_i = 0
            else:
                current_episode_i += 1

            if current_llm_i == len(llms) - 1:
                current_llm_i = 0
            
            #iterate first through all titles and then llms
            # episode_title = episode_titles_to_choose_from[current_episode_i]
            # llm = llms[current_llm_i]
            llm = random.choice(llms)
            episode_title = random.choice(episode_titles_to_choose_from)
            
            
            start_time = time.time()  # Start timer

            episode_identifier = sanitize_filename(f"{llm}_{episode_title}")

            WIP_path = f"./WIP_episode/{episode_identifier}"  # Clean working folder
            if os.path.exists("./WIP_episode/"):
                shutil.rmtree("./WIP_episode/")
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
                print(
                    f"\033[38;5;214mGenerating voices: {i+1}/{len(episode.actions)}\033[0m"
                )
                
            
            
            # move Episode from WIP to ready
            episode_version = 0
            while os.path.exists(f"./shared/StreamingAssets/unreleased_episodes/{episode_version}_{episode_identifier}") or os.path.exists(f"./shared/StreamingAssets/prioritized_episodes/{episode_version}_{episode_identifier}") or os.path.exists(f"./shared/StreamingAssets/released_episodes/{episode_version}_{episode_identifier}"):
                episode_version += 1
            generated_episode_folder = f"./shared/StreamingAssets/prioritized_episodes/{episode_version}_{episode_identifier}"
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
                print(
                    f"\033[38;5;255mAverage Time for {llm_name} to produce actions: {average_time:.0f} seconds\033[0m"
                )
                
        except Exception as e:
            print("\033[91mAn error occurred:\033[0m", e)
            
            # Print file name and line number
            tb = traceback.extract_tb(e.__traceback__)
            filename, line, func, text = tb[-1]
            print(f"\033[93mFile: {filename}, Line: {line}, In: {func}\033[0m")

            # Separately print the full call stack
            print("\033[94mCall Stack:\033[0m")
            print("".join(traceback.format_tb(e.__traceback__)))
            raise(e)
            time.sleep(1)



@profile
def set_supported_scenes():
    global supported_scenes
    # Reading the JSON data from the specified file
    with open("./shared/supported_scenes.json", 'r') as file:
        file_data = file.read()

    print(file_data)
    supported_scenes = SupportedScenes.from_json(file_data)
    print("Supported scenes set successfully!")
    
@profile
def augment_generated_episodes():
    all_dirs = list(os.walk("./shared/StreamingAssets/"))
    shuffle(all_dirs)
    for dirpath, dirnames, filenames in all_dirs:
        if "actions.json" in filenames:
            json_path = os.path.join(dirpath, "actions.json")

            with open(json_path, 'r') as file:
                json_content = json.load(file)
            start_time = time.time()  # Record start time
            try:
                episode = Episode.from_json(json.dumps(json_content))
            except Exception as e:
                print(f"Error processing episode: {e}")
                continue

            elapsed_time = time.time() - start_time  # Calculate elapsed time
            if elapsed_time > 10 and "StreamingAssets/released_episodes" in dirpath:
                # Move the entire folder if processing took longer than 10 seconds
                new_dirpath = dirpath.replace("released_episodes", "unreleased_episodes")
                shutil.move(dirpath, new_dirpath)
                # Update actions.json in the new location
                new_json_path = os.path.join(new_dirpath, "actions.json")
                with open(new_json_path, 'w') as file:
                    file.write(episode.to_json())
            else:
                # Overwrite the original actions.json if processing was successful and within time limit
                with open(json_path, 'w') as file:
                    file.write(episode.to_json())



# augment_generated_episodes()
generate_episodes()