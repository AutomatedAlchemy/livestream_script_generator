import asyncio
import base64
import glob
import json
import os
import re
from typing import List

import aiohttp
from line_profiler import profile

from classes.Action import Action
from classes.cls_web_scraper import WebScraper
from classes.DisplayableContent import DisplayableContent
from classes.Location import Location
from interface.cls_few_shot_factory import FewShotProvider
from interface.cls_ollama_client import OllamaClient


def try_json_to_object(json_string: str) -> object:
    json_string = json_string.replace("'''json", "").replace("'''", "")
    json_string = json_string.replace("'''json", "").replace("'''", "")
    json_string = json_string.replace(": False", ": false").replace(": True", ": true")
    json_string = json_string.replace("''", '""')
    last_bracket_index = json_string.rfind("]")
    if last_bracket_index != -1:
        try:
            return json.loads(json_string[: last_bracket_index + 1])
        except:
            pass
    else:
        last_curly_index = json_string.rfind("}")
        if last_curly_index != -1:
            try:
                return json.loads(json_string[: last_curly_index + 1] + "]")
            except:
                pass
    last_quote_index = json_string.rfind('"')
    if last_quote_index != -1:
        try:
            return json.loads(json_string[: last_quote_index + 1] + "}]")
        except:
            pass
    return None


def extract_script(script_text: str) -> str:
    if script_text.count("'''python") == 1:
        script_text = script_text.split("'''python")[1]
    if script_text.count("'''") == 2:
        script_text = script_text.split("'''")[1]
    if script_text.count("'''") == 1:
        script_text = script_text.split("'''")[0]
    if script_text.count("```python") == 1:
        script_text = script_text.split("```python")[1]
    if script_text.count("```") == 2:
        script_text = script_text.split("```")[1]
    if script_text.count("```") == 1:
        script_text = script_text.split("```")[0]

    import_index = script_text.find("import")
    if import_index != -1:  # Check if "import" is found
        script_text = script_text[import_index:]
    else:
        script_text = ""  # or handle the case where "import" is not found as needed

    return script_text.strip().strip("\n").strip().strip("\n").strip().strip("\n")


class Episode:
    def __init__(
        self,
        show_title: str,
        episode_title: str,
        characters: List[str],
        location: Location,
        outline: str = "",
        actions: List[Action] = [],
        llm: str = "orca2",
        displayable_content: DisplayableContent = None,
        load_only: bool = False,
        loop: asyncio.AbstractEventLoop = asyncio.new_event_loop(),
    ):
        """
        :param outline: Outline for the episode. If no outline is provided, it will be generated automatically.
        :param actions: Actions for the episode. If no actions is provided, it will be generated automatically.
        :param displayable_content: base64 Image.
        """
        self.loop = loop
        asyncio.set_event_loop(self.loop)
        self.llm = llm
        self.show_title = show_title
        self.episode_title = episode_title
        self.characters = characters
        self.location = location
        self.session = OllamaClient()  # Assuming OllamaClient is defined elsewhere
        self.outline: str = outline
        self.actions: List[Action] = actions
        if displayable_content:
            self.displayable_content: DisplayableContent = displayable_content
        else:
            self.displayable_content: DisplayableContent = DisplayableContent()

        if load_only:
            return

        if self.displayable_content.to_json() == DisplayableContent().to_json():
            self.generate_displayableContent(self.episode_title)

        if self.actions and not self.outline:
            self.outline = self.session.generate_completion(
                f"Please author an outline of the following episode script of the show '{self.show_title}' script: '''json\n{json.dumps([action.to_json() for action in self.actions])}\n'''",
                llm,
                "Sure! In this episode",
            )
        if len(self.actions) == 0:
            self.generate_actions()

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
    def from_json(cls, json_str: str, load_only: bool = False):
        data: dict = json.loads(json_str)

        location = Location.from_json(data["location"])
        actions = [Action.from_json(action) for action in data["actions"]]
        displayable_content = data.get("displayable_content")
        if displayable_content:
            displayable_content = DisplayableContent.from_json(displayable_content)
        # Constructing the Episode instance
        episode = cls(
            data["show_title"],
            data["episode_title"],
            data["characters"],
            location,
            data.get("outline"),
            actions,
            displayable_content=displayable_content,
            load_only=load_only,
        )
        return episode

    @profile
    def generate_displayableContent(self, topic: str) -> None:
        def try_web_image_search() -> str:
            search_term = topic

            scraper = WebScraper()
            self.loop.run_until_complete(
                scraper.initialize_image_fetcher(search_term, 100)
            )


            def image_fits_topic(base64_image: str) -> bool:
                image_description: str = self.session.generate_completion(
                    f"What is shown in the image?",
                    "bakllava:7b-v1-q4_K_M",
                    images=[base64_image],
                )
                i: int = len([f for f in os.listdir("./test/")])
                with open(f"./test/{str(i)}.jpg", "wb") as file:
                    file.write(base64.b64decode(base64_image))
                if not image_description:
                    return False

                is_topic_appropriate_response = (
                    FewShotProvider.few_shot_isImageTopicAppropriate(
                        self.episode_title, image_description, self.llm
                    )
                )

                print("\033[92m" + "SEARCHED KEYWORD: " + search_term + "\033[0m")
                print("\033[92m" + "TOPIC: " + topic + "\033[0m")
                print("image_content_response: " + image_description)
                print("is_topic_appropriate_response: " + is_topic_appropriate_response)

                return "yes" in is_topic_appropriate_response.lower()

            os.makedirs("./test", exist_ok=True)
            fitting_image_base64 = self.loop.run_until_complete(
                scraper.get_next_image_as_base64(image_fits_topic)
            )

            if fitting_image_base64:
                with open(f"./test/true_image.jpg", "wb") as file:
                    file.write(base64.b64decode(fitting_image_base64))
                image_description: str = self.session.generate_completion(
                    f"What is shown in the image?",
                    "bakllava:7b-v1-q4_K_M",
                    images=[fitting_image_base64],
                )
                image_title: str = self.session.generate_completion(
                    f"Come up with a title for the image. Here is the image description: '{image_description}'",
                    self.llm,
                    "Sure! A fitting title would be: 'Image of ",
                ).split("'")[1]

                return fitting_image_base64, image_title
            else:
                print("\033[91m Did not find any appropriate image on the web! \033[0m")
                raise (Exception("This seems like an issue?"))
            return "", ""

        # @profile
        # def try_python_visualization():
        #     local_vars = {
        #         "base64": base64,
        #         "BytesIO": BytesIO,
        #         "np": np,
        #         "plt": plt,
        #     }
        #     generated_script_response = FewShotProvider.few_shot_generatePythonScript(
        #         self.episode_title
        #     )
        #     generated_script: str = extract_script(generated_script_response)
        #     error: str = ""
        #     try:
        #         generated_script.replace(
        #             "import base64\nfrom io import BytesIO\nimport numpy as np\nimport matplotlib.pyplot as plt\n",
        #             "",
        #         )
        #         exec(generated_script, globals())
        #         base64_image = local_vars["base64_image"]
        #         print(f"GENERATED IMAGE:\n\n{base64_image}\n")
        #         self.displayable_content.blackboard_graph = base64_image
        #     except Exception as e:
        #         error = str(e)
        #         if "base64_image" in error:
        #             error = "ERROR: MISSING GLOBAL VARIABLE: 'base64_image'"
        #         retry_count = 0
        #         max_retries = 5
        #         chat_error_fixing: Chat = Chat(
        #             "You are GPT-5. As a generalist expert with superhuman coding abilities you have been trained to always comply with the users requests. You always reimplement python scripts fully, expertly and flawlessly. Your scripts always exit only after having generated the global base64_image variable."
        #         )
        #         chat_error_fixing.add_message(
        #             Role.USER,
        #             f"Please implement a python script to visualize the topic of '{self.episode_title}'. The script must exit only after the visualization has been put into a global variable called 'base64_image', as this variable will be accessed externally.",
        #         )
        #         chat_error_fixing.add_message(
        #             Role.ASSISTANT,
        #             f"Sure! Here's a script which will visualize the topic of '{self.episode_title}'.\n'''python\n{generated_script}\n'''",
        #         )
        #         while retry_count < max_retries:
        #             try:
        #                 print(f"{error}")
        #                 chat_error_fixing.add_message(
        #                     Role.USER,
        #                     f"The script you implemented threw an error, think step by step about it's cause and provide the reimplemented script in full, afterwards.\n'''error\n{error}\n'''",
        #                 )
        #                 generated_script_response: str = (
        #                     self.session.generate_completion(
        #                         chat_error_fixing, "wizardcoder:7b-python", debug=True
        #                     )
        #                 )
        #                 chat_error_fixing.add_message(
        #                     Role.ASSISTANT, generated_script_response
        #                 )

        #                 generated_script = extract_script(generated_script_response)

        #                 generated_script.replace(
        #                     "import base64\nfrom io import BytesIO\nimport numpy as np\nimport matplotlib.pyplot as plt\n",
        #                     "",
        #                 )
        #                 exec(generated_script, globals(), local_vars)
        #                 base64_image = local_vars["base64_image"]
        #                 print(f"GENERATED IMAGE:\n\n{base64_image}\n")
        #                 return base64_image
        #             except Exception as e:
        #                 error = str(e)
        #                 if "base64_image" in error:
        #                     error = "ERROR: MISSING GLOBAL VARIABLE: The script requires the following global variable: 'base64_image'"
        #                 retry_count += 1
        #                 if retry_count >= max_retries:
        #                     print(
        #                         f"Error after {max_retries} attempts, quitting...: {e}"
        #                     )
        #                     break
        #                 else:
        #                     print(f"Attempt {retry_count} failed, retrying...: {e}")

        # python_visualization = try_python_visualization()

        # if (python_visualization):
        #     self.displayable_content = python_visualization
        #     return
        scraped_visualization, image_title = try_web_image_search()
        if scraped_visualization:
            self.location.interactableObjects.append(
                "Blackboard image of: " + image_title
            )
            self.displayable_content.blackboard_image = scraped_visualization

            blackboard_caption = FewShotProvider.few_shot_generateBlackboardCaption(
                topic, image_title, self.llm
            )
            blackboard_caption.split("'''chalkboard_caption\n")[1].split("'''")[
                0
            ].strip("\n").strip().strip("\n").strip()
            if len(blackboard_caption) > 500:
                blackboard_caption: str = self.session.generate_completion(
                    f"Please boil down the following blackboard caption for use on a smaller blackboard: '{blackboard_caption}'.",
                    self.llm,
                    "Sure!\n'''chalkboard_caption\n",
                )
                blackboard_caption.split("'''chalkboard_caption\n")[1].split("'''")[
                    0
                ].strip("\n").strip().strip("\n").strip()

            self.displayable_content.blackboard_caption = blackboard_caption

    @profile
    def generate_actions(self, regenerate_outline: bool = False) -> List[Action]:
        if (not self.outline) or regenerate_outline:
            self.outline: str = FewShotProvider.few_shot_topicToEpisodeOutline(
                self.episode_title,
                self.characters,
                self.location,
                self.llm,
                self.show_title,
            )
        temperature = 0.95
        actions: List[Action] = None
        while not actions:
            actions_str = FewShotProvider.few_shot_outlineToActions(
                self.outline, self.llm, temperature
            )
            actions: List[Action] = try_json_to_object(actions_str)
            temperature -= 0.1
            if not actions:
                print(
                    f"\033[91mWarning: Received json actions has invalid format. Adjusting temperature to {temperature} and retrying...\033[0m"
                )
                if temperature <= 0.1:
                    return self.generate_actions(
                        True
                    )  # The outline seems to break the few_shot_outlineToActions method, let's generate a new outline an retry
        self.actions = [Action.from_json(action) for action in actions]
