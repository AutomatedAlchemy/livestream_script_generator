import json
from typing import List, Optional

from classes.Action import Action
from classes.DisplayableContent import DisplayableContent
from classes.Location import Location


class struct_Episode:
    def __init__(
        self,
        show_title: str,
        episode_title: str,
        characters: List[str],
        location: Location,
        outline: str = "",
        actions: List[Action] = [],
        llm: str = "orca2",
        displayable_content: Optional[DisplayableContent] = None,
    ):
        """
        :param outline: Outline for the episode. If no outline is provided, it will be generated automatically.
        :param actions: Actions for the episode. If no actions is provided, it will be generated automatically.
        :param displayable_content: base64 Image.
        """
        self.llm = llm
        self.show_title = show_title
        self.episode_title = episode_title
        self.characters = characters
        self.location = location
        self.outline: str = outline
        self.actions: List[Action] = actions
        self.displayable_content: DisplayableContent
        if displayable_content:
            self.displayable_content = displayable_content
        else:
            self.displayable_content = DisplayableContent()

    def to_json(self) -> str:
        # Convert the object to a JSON string
        return json.dumps(
            {
                "show_title": self.show_title,
                "episode_title": self.episode_title,
                "characters": self.characters,
                "displayable_content": self.displayable_content.to_json(),
                "location": self.location.to_json(),
                "actions": [action.to_json() for action in self.actions],
            },
            indent=4,
        )

    @classmethod
    def from_json(cls, json_str: str):
        data: dict = json.loads(json_str)

        location = Location.from_json(data["location"])
        actions = [Action.from_dict(action) for action in data["actions"]]
        displayable_content = data.get("displayable_content")
        if displayable_content:
            if isinstance(displayable_content, dict):
                displayable_content = json.dumps(displayable_content)
            displayable_content = DisplayableContent.from_json(displayable_content)
        # Constructing the Episode instance
        episode = cls(
            data["show_title"],
            data["episode_title"],
            data["characters"],
            location,
            str(data.get("outline")),
            actions,
            displayable_content=displayable_content,
        )
        return episode