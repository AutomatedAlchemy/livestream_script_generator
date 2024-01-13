import json
from typing import List


class Location:
    def __init__(self, title: str, interactableObjects: List[str]):
        self.title = title
        self.interactableObjects = interactableObjects

    def to_json(self) -> str:
        return json.dumps({'title': self.title, 'interactableObjects': self.interactableObjects})

    @classmethod
    def from_json(cls, json_str: str) -> 'Location':
        data = json.loads(json_str)
        return cls(data['title'], data['interactableObjects'])
