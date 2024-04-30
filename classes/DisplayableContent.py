import json


class DisplayableContent:
    def __init__(
        self,
        blackboard_caption: str = "",
        blackboard_image: str = "",
        blackboard_graph: str = "",
    ):
        self.blackboard_caption: str = blackboard_caption
        self.blackboard_image: str = blackboard_image
        self.blackboard_graph: str = blackboard_graph

    def to_json(self):
        """Convert instance to JSON string."""
        return self.__dict__

    @classmethod
    def from_json(cls, json_str):
        """Create instance from JSON string."""
        attributes = json.loads(json_str)
        return cls(**attributes)
