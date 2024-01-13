
import json
import os
import random
from typing import List, Tuple

from classes.Action import Action
from classes.Location import Location
from interface.cls_chat import Chat, Role
from interface.cls_ollama_client import OllamaClient


def is_valid_json(json_string: str) -> str | None:
    json_string = json_string.replace("'''json", "").replace("'''", "")
    json_string = json_string.replace(": False", ": false").replace(": True", ": true")
    last_bracket_index = json_string.rfind(']')
    if last_bracket_index != -1:
        try:
            return json.loads(json_string[:last_bracket_index + 1])
        except:
            pass
    else:
        last_curly_index = json_string.rfind('}')
        if last_curly_index != -1:
            try:
                return json.loads(json_string[:last_curly_index + 1]+"]")
            except:
                pass
    last_quote_index = json_string.rfind('"')
    if last_quote_index != -1:
        try:
            return json.loads(json_string[:last_quote_index + 1] + "}]")
        except:
            pass
    return None

class Episode:
    def __init__(
        self,
        show_title: str,
        episode_title: str,
        characters: List[str],
        location: Location,
        outline: str = "",
        script: List[Action] = [],
        llm: str = "zephyr"
    ):
        """
        :param script: Outline for the episode. If no outline is provided, it will be generated automatically.
        :param script: Script for the episode. If no script is provided, it will be generated automatically.
        """
        self.llm = llm
        self.show_title = show_title
        self.episode_title = episode_title
        self.characters = characters
        self.location = location
        self.session = OllamaClient()  # Assuming OllamaClient is defined elsewhere
        self.outline: str
        if (script and not outline):
            self.outline = self.session.generate_completion(f"Please author an outline of the following episode script of the show '{self.show_title}' script: '''json\n{json.dumps([action.to_json() for action in self.script])}\n'''", llm, "Sure! In this episode")
        else:
            self.outline = outline
        self.script: List[Action]
        if len(script) == 0:
            self.script = self.generate_script()
        else:
            self.script = script

    def to_json(self) -> str:
        # Convert the object to a JSON string
        return json.dumps(
            {
                "show_title": self.show_title,
                "episode_title": self.episode_title,
                "characters": self.characters,
                "location": self.location.to_json(),  # Assuming Location has a to_json method
                "script": [
                    action.to_json() for action in self.script
                ],  # Assuming Action has a to_json method
            },
            indent=4,
        )

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)

        location = Location.from_json(json.dumps(data["location"]))
        script = [Action.from_json(action) for action in json.loads(data["script"])]
        
        # Constructing the Episode instance
        episode = cls(
            data["show_title"], data["episode_title"], data["characters"], location, data["outline"], script
        )
        return episode

    def get_script_and_json_few_shot_chat(self) -> Tuple[Chat, Chat]:
        global few_shot_episodes
        episode_script_few_shot_chat = Chat(
            f'You are an expert script writer who has dedicated his life to studying all fields of science to condense his knowledge into a show named: "{self.show_title}", to bring accurate and factual knowledge to everyone for free!.'
        )
        script_to_json_few_shot_chat = Chat(
            "Convert the provided story into the given standardized json format. Make the script scientific, engaging and thought-provoking."
        )
        for episode in few_shot_episodes:
            episode_script_few_shot_chat.add_message(
                Role.USER,
                f'Hi, please come up with an episode of "{episode.show_title}" revolving around the topic of "{episode.episode_title}", involving the characters {", ".join(episode.characters)} at the location "{episode.location.title}"',
            )
            episode_script_few_shot_chat.add_message(Role.ASSISTANT, episode.outline)

            script_to_json_few_shot_chat.add_message(Role.USER, episode.outline)
            script_to_json_few_shot_chat.add_message(
                Role.ASSISTANT, f"'''json\n{[action.to_json() for action in episode.script]}\n'''"
            )

        return episode_script_few_shot_chat, script_to_json_few_shot_chat

    def generate_script(self) -> List[Action]:
        (
            episode_script_few_shot_chat,
            script_to_json_few_shot_chat,
        ) = self.get_script_and_json_few_shot_chat()

        if (not self.outline):
            episode_script_few_shot_chat.add_message(
                Role.USER,
                f'Hi, please come up with an outline for a short episode of "{self.show_title}", this episodes title will be "{self.episode_title}", staged at the location "{self.location.title}" and populated by the characters "{self.characters}". The location contains {self.location.interactableObjects} with which the characters can interact with.',
            )

            self.outline = self.session.generate_completion(
                episode_script_few_shot_chat,
                self.llm,
                f'Sure! In this short episode of "{self.show_title}" appropriately named "{self.episode_title}" staged by the characters "{self.characters}" in the location "{self.location.title}"',
            )
            print(self.outline)
            script_to_json_few_shot_chat.add_message(Role.USER, self.outline)

        temperature = 0.85
        start_response_with = (
            """'''json\n[\n\t{\n\t\t"character": \""""
            + random.choice(self.characters)
            + '",\n\t\t"voice_line": "'
        )

        json_script = self.session.generate_completion(
            script_to_json_few_shot_chat,
            self.llm,
            start_response_with,
            temperature=temperature,
        )
        print(json_script)
        actions = is_valid_json(json_script)
        while not actions:
            temperature -= 0.1
            if temperature <= 0.1:
                return self.generate_script()
            print(
                f"\033[91mWarning: Received json script has invalid format. Adjusting temperature to {temperature} and retrying...\033[0m"
            )
            json_script = self.session.generate_completion(
                script_to_json_few_shot_chat,
                self.llm,
                start_response_with,
                temperature=temperature,
            )

            print(json_script)
            json_script = json_script.split("'''json")[1].split("'''")[0]
            actions = is_valid_json(json_script)

        return [Action.from_json(action) for action in actions]
    
few_shot_episodes: List[Episode] = []
for episode_path in os.listdir("./few_shot_episodes/"):
    with open(os.path.join("./few_shot_episodes/", episode_path), "r") as file:
        episode_data = json.load(file)
        episode_json_str = json.dumps(
            episode_data
        )  # Convert the dictionary to a JSON string
        few_shot_episodes.append(Episode.from_json(episode_json_str))