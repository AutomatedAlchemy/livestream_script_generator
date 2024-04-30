import json
from typing import List


class Location:
    def __init__(self, title: str, interactableObjects: List[str]):
        self.title = title
        self.interactableObjects = interactableObjects

    def to_json(self) -> dict:
        return self.__dict__

    @classmethod
    def from_json(cls, json_str: str) -> 'Location':
        data: dict
        if (isinstance(json_str, str)):
            data = json.loads(json_str)
        elif (isinstance(json_str, dict)):
            data = json_str
        return cls(data['title'], data['interactableObjects'])
