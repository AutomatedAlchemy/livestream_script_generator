import json
import math
import os
from enum import Enum
from typing import Dict, List, Tuple

from colorama import Fore, Style
from jinja2 import Template


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Chat:
    def __init__(self, instruction_message: str = ""):
        self.messages: List[Tuple[Role, str]] = []
        if instruction_message:
            self.add_message(Role.SYSTEM, instruction_message)

    def add_message(self, role: Role, content: str):
        self.messages.append((role, content))

    def __getitem__(self, key):
        return self.messages[key]

    def __str__(self):
        return json.dumps(
            [
                {"role": message[0].value, "content": message[1]}
                for message in self.messages
            ]
        )

    def to_openai_chat(self) -> List[Dict[str, str]]:
        return [
            {"role": message[0].value, "content": message[1]}
            for message in self.messages
        ]

    def to_oobabooga_history(self) -> Tuple[Dict[str, List[List[str]]], str]:
        internal_arr: List[List[str]] = []
        visible_arr: List[List[str]] = []
        instruction: str = ""

        for i, (role, content) in enumerate(self.messages):
            if role == Role.SYSTEM:
                instruction = content if i == 0 else instruction
            elif role in [Role.USER, Role.ASSISTANT]:
                pair = [content, ""] if role == Role.USER else ["", content]
                internal_arr.append(pair)
                visible_arr.append(pair)

        return {"internal": internal_arr, "visible": visible_arr}, instruction

    def to_jinja2(self, template_str: str) -> str:
        # Create a template from the corrected string
        template = Template(template_str)
        formatted_message = ""
        # {"system": self.message[0].value, "prompt": self.message[1].value}
        for i in range(math.ceil(len(self.messages) / 2)):
            # role: str = i % 2 == 0 if "human" else "assistant"
            formatted_message += template.render(
                {
                    "system": self.messages[i * 2][1],
                    "prompt": self.messages[i * 2 + 1][1],
                }
            )

        return formatted_message

    def print_chat(self):
        """
        Print the chat messages with colored roles.
        """
        for role, content in self.messages:
            if isinstance(role, Role):
                role_value = role.value
            else:
                role_value = role  # Assuming it's a string
            if role == Role.ASSISTANT or role == Role.SYSTEM:
                formatted_content = f"{Fore.BLUE}{content}{Style.RESET_ALL}"
                print(f"{formatted_content} :{role_value}")
            else:
                formatted_role = f"{Fore.GREEN}{role_value}{Style.RESET_ALL}"
                print(f"{formatted_role}: {content}")

    def save_session(self, file_path: str = "./cache/chat_session_main.json"):
        """Save the Chat instance to a JSON file."""
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Convert the Chat object to a dictionary and save it as JSON
        with open(file_path, "w") as file:
            json.dump(self.to_dict(), file, indent=4)

    @classmethod
    def load(cls, file_path):
        """Load a Chat instance from a JSON file."""
        with open(file_path, "r") as file:
            data_dict = json.load(file)
        return cls.from_dict(data_dict)

    def to_dict(self):
        return {
            "messages": [
                {"role": role.value, "content": content}
                for role, content in self.messages
            ]
        }

    @classmethod
    def from_dict(cls, data_dict):
        chat_instance = cls()
        for message in data_dict["messages"]:
            role = Role[message["role"].upper()]
            content = message["content"]
            chat_instance.add_message(role, content)
        return chat_instance
