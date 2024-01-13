class Action:
    def __init__(
        self,
        character: str,
        voice_line: str = "",
        looking_at: str = "",
        walking_to: bool = False,
    ):
        self.character = character
        self.voice_line = voice_line
        self.looking_at = looking_at
        self.walking_to = walking_to

    def to_json(self):
        return {
            "character": self.character,
            "voice_line": self.voice_line,
            "looking_at": self.looking_at,
            "walking_to": self.walking_to,
        }

    @classmethod
    def from_json(cls, action_data: dict):
        return cls(
            character=action_data["character"],
            voice_line=action_data.get("voice_line", ""),
            looking_at=action_data.get("looking_at", ""),
            walking_to=action_data.get("walking_to", False),
        )