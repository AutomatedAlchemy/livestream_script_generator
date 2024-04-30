import json
import os
import random
from typing import Dict, List, Union

from classes.struct_Episode import struct_Episode


class PollOption:
    def __init__(self, letter: str, votes: int, episode_title: str) -> None:
        self.letter: str = letter
        self.votes: int = votes
        self.episode_title: str = episode_title

    def to_dict(self) -> Dict[str, Union[str, int]]:
        return {"letter": self.letter, "votes": self.votes, "episode_title": self.episode_title}

    @staticmethod
    def from_dict(data: Dict[str, Union[str, int]]) -> 'PollOption':
        return PollOption(
            letter=str(data['letter']), 
            votes=int(data['votes']),  # Explicitly convert votes to int
            episode_title=str(data['episode_title'])
        )


class Poll:
    def __init__(self, next_episode_options: List[str] = []) -> None:
        self.pollOptions: List[PollOption] = []
        if next_episode_options:
            chosen_episode_options: List[str] = random.choices(next_episode_options, k=3)
            self.pollOptions.append(PollOption(letter="A", votes=0, episode_title=self.get_title(chosen_episode_options[0])))
            self.pollOptions.append(PollOption(letter="B", votes=0, episode_title=self.get_title(chosen_episode_options[1])))
            self.pollOptions.append(PollOption(letter="C", votes=0, episode_title=self.get_title(chosen_episode_options[2])))
    
    def get_title(self, episode_folder_path:str):
        with open(episode_folder_path + "/actions.json", "r") as file:
            episode_data = file.read()
        episode: struct_Episode = struct_Episode.from_json(episode_data)
        return episode.episode_title
        
        
    def to_json(self) -> str:
        poll_options_list: List[Dict[str, Union[str, int]]] = [option.to_dict() for option in self.pollOptions]
        return json.dumps({"pollOptions": poll_options_list}, indent=4)

    def update_votes(self, message: str) -> None:
        for option in self.pollOptions:
            if message == option.letter:
                option.votes += 1

    def to_file(self) -> None:
        with open("./cache/poll_votes.json", 'w') as file:
            file.write(self.to_json())

    @classmethod
    def from_file(cls) -> 'Poll':
        with open("./cache/poll_votes.json", 'r') as file:
            data = json.load(file)
            poll = cls()
            poll.pollOptions = [PollOption.from_dict(option) for option in data["pollOptions"]]
            return poll
