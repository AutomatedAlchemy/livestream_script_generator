import json
from typing import List

from classes.Location import Location

class SupportedScenes:
    def __init__(self, characters: List[str], locations: List[Location]):
        self.characters = characters
        self.locations = locations

    def to_json(self) -> str:
        # Serialize the object to a JSON string
        return json.dumps(
            {
                "characters": self.characters,
                "locations": [location.to_json() for location in self.locations]
            },
            indent=4
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'SupportedScenes':
        data = json.loads(json_str)
        return cls(data['characters'], [Location.from_json(location_json) for location_json in data['locations']])
