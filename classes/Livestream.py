import random
from typing import List

from classes.Action import Action
from classes.Episode import Episode
from classes.SupportedScenes import SupportedScenes


class Livestream:
    def __init__(self, title: str):
        self.title = title

    def random_selection(self, my_list):
        if not my_list:  # Check if the list is empty
            return []

        num_to_select = random.randint(1, len(my_list))  # Number of elements to select
        return random.sample(my_list, num_to_select)

    def generate_episode(self, episode_title: str, supported_scenes: SupportedScenes, llm:str = "orca2") -> Episode:
        episode = Episode(
            self.title,
            episode_title,
            supported_scenes.characters,
            # self.random_selection(supported_scenes.characters),
            random.choice(supported_scenes.locations),
            llm=llm
        )

        return episode