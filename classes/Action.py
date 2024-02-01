import base64
from io import BytesIO
from typing import Optional

from PIL import Image


class Action:
    def __init__(
        self,
        character: str,
        voice_line: str = "",
        looking_at: str = "",
        walking_to: str = "",
    ):
        self.character = character
        self.voice_line = voice_line
        self.looking_at = looking_at
        self.walking_to = walking_to

    def to_json(self):
        return self.__dict__

    @classmethod
    def from_json(cls, action_data: dict):
        return cls(
            character=action_data["character"],
            voice_line=action_data.get("voice_line", ""),
            looking_at=action_data.get("looking_at", ""),
            walking_to=action_data.get("walking_to", ""),
        )

    # def get_image(self) -> Optional[Image.Image]:
    #     """Returns the PIL Image object from the base64 string if it exists."""
    #     if self.displayable_content:
    #         image_data = base64.b64decode(self.displayable_content)
    #         return Image.open(BytesIO(image_data))
    #     return None

    # def set_image(self, image: Image.Image):
    #     """Sets the displayable content from a PIL Image object."""
    #     buffer = BytesIO()
    #     image.save(buffer, format="PNG")
    #     self.displayable_content = base64.b64encode(buffer.getvalue()).decode()
