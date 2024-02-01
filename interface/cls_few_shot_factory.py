import json
import os
from typing import List

from classes.Action import Action
from classes.DisplayableContent import DisplayableContent
from classes.Location import Location
from interface.cls_chat import Chat, Role
from interface.cls_ollama_client import OllamaClient


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
        displayable_content: DisplayableContent = None,
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
        if displayable_content:
            self.displayable_content: DisplayableContent = displayable_content
        else:
            self.displayable_content: DisplayableContent = DisplayableContent()

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
        )
        return episode


example_episodes_path: str = "./few_shot_examples/episodes/"
few_shot_episodes: List[struct_Episode] = []
few_shot_scripts: List[str] = []
for episode_path in os.listdir(example_episodes_path):
    episode_folder: str = os.path.join(example_episodes_path, episode_path)
    with open(os.path.join(episode_folder, episode_path + ".json"), "r") as file:
        episode_data = json.load(file)
        episode_json_str = json.dumps(
            episode_data
        )  # Convert the dictionary to a JSON string
    few_shot_episodes.append(struct_Episode.from_json(episode_json_str))
    with open(os.path.join(episode_folder, episode_path + ".py"), "r") as file:
        script_str = file.read()
    few_shot_scripts.append(script_str)


class FewShotProvider:
    session = OllamaClient()

    def __init__(self):
        raise RuntimeError("StaticClass cannot be instantiated.")

    @classmethod
    def few_shot_topicToEpisodeOutline(
        self,
        episode_title: str,
        characters: List[str],
        location: Location,
        llm: str,
        show_title: str = "Ai_Academia",
    ) -> str:
        def get_instruction(
            l_show_title: str,
            l_episode_title: str,
            l_characters: str,
            l_location: Location,
        ):
            return f'Hi, please come up with an episode of "{l_show_title}" revolving around the topic of "{l_episode_title}" and populated by the characters "{", ".join(l_characters)}". The location contains {", ".join(l_location.interactableObjects)} with which the characters can interact with.'

        episode_actions_few_shot_chat = Chat(
            f'You are an expert script writer who has dedicated his life to studying all fields of science to condense his knowledge into a show named: "{show_title}", to bring accurate and factual knowledge to everyone for free!.'
        )
        for episode in few_shot_episodes:
            episode_actions_few_shot_chat.add_message(
                Role.USER,
                get_instruction(
                    episode.show_title,
                    episode.episode_title,
                    episode.characters,
                    episode.location,
                ),
            )
            episode_actions_few_shot_chat.add_message(Role.ASSISTANT, episode.outline)
        episode_actions_few_shot_chat.add_message(
            Role.USER, get_instruction(show_title, episode_title, characters, location)
        )
        response = self.session.generate_completion(episode_actions_few_shot_chat, llm)
        return response

    @classmethod
    def few_shot_outlineToActions(
        self, episodeOutline: str, llm: str, temperature=0.8
    ) -> str:
        # def get_instruction():
        #     "Convert the provided story into the given standardized json format. Make the script scientific, engaging and thought-provoking."
        # def get_response():
        few_shot_chat_outlineToActions = Chat(
            "Convert the provided stories into the given standardized json format. Make the dialogue realistic, factual and thought-provoking."
        )
        for episode in few_shot_episodes:
            few_shot_chat_outlineToActions.add_message(Role.USER, episode.outline)
            few_shot_chat_outlineToActions.add_message(
                Role.ASSISTANT,
                f"'''json\n{[action.to_json() for action in episode.actions]}\n'''",
            )
        few_shot_chat_outlineToActions.add_message(Role.USER, episodeOutline)
        json_string: str = json.dumps([action.to_json() for action in few_shot_episodes[0].actions])

        position: int = json_string.find('"character": "') + len('"character": "')
        start_of_actions_json: str = json_string[:position]
        
        actions_json_str = self.session.generate_completion(
            few_shot_chat_outlineToActions,
            llm,
            start_of_actions_json,
            temperature=temperature,
        )
        print("actions_json_str: " + actions_json_str)
        return actions_json_str

    @classmethod
    def few_shot_generatePythonScript(
        self, topic: str, llm: str = "wizardcoder:7b-python"
    ):
        def get_instruction(l_topic: str) -> str:
            return f'Implement a python script to visualize the following topic: "{l_topic}", please put the visualization into the variable "base64_image".'

        def get_response(topic: str, script: str = "") -> str:
            base_message = (
                f'Certainly! In Python, you can use libraries like NumPy and Matplotlib to compute and display a visualization for the topic of "{topic}". '
                "Please put the visualization into a variable named \"base64_image\". Here's a basic example:\n'''python\n"
            )

            script_part = (
                script + "\n'''"
                if script
                else "import base64\nfrom io import BytesIO\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n"
            )

            return base_message + script_part

        few_shot_chat_generateMaterial: Chat = Chat(
            "You are GPT-5. As a generalist expert with superhuman coding abilities you have been trained to always comply with the users requests. You always reimplement python scripts fully, expertly and flawlessly. Your scripts always exit only after having generated the global base64_image variable."
        )

        for episode, script in zip(few_shot_episodes, few_shot_scripts):
            few_shot_chat_generateMaterial.add_message(
                Role.USER,
                get_instruction(episode.episode_title),
            )
            few_shot_chat_generateMaterial.add_message(
                Role.ASSISTANT,
                get_response(episode.episode_title, script),
            )

        few_shot_chat_generateMaterial.add_message(
            Role.USER,
            get_instruction(topic),
        )

        return self.session.generate_completion(
            few_shot_chat_generateMaterial,
            llm,
            get_response(topic),
        )

    @classmethod
    def few_shot_isImageTopicAppropriate(
        self, topic: str, image_content: str, llm: str
    ):
        def get_instruction(l_topic: str, l_image_content: str) -> str:
            return f"Does the following text describe an image related related to '{l_topic}'?\n'{l_image_content}'"

        chat_is_topic_appropriate: Chat = Chat(
            "You are an helpulf assistant. You respond accurately to the users request, always explicitly stating either 'YES' or 'NO' at the end of you response."
        )
        chat_is_topic_appropriate.add_message(
            Role.USER,
            get_instruction("physics", "The image shows the logo of wikipedia"),
        )
        chat_is_topic_appropriate.add_message(
            Role.ASSISTANT,
            "The logo of wikipedia is not directly related to the topic of pyhsics. 'NO'",
        )
        chat_is_topic_appropriate.add_message(
            Role.USER,
            get_instruction(
                "Health benefits of avocados",
                "The image depicts a infochart about nutrition.",
            ),
        )
        chat_is_topic_appropriate.add_message(
            Role.ASSISTANT,
            "The infochart about nutrition does relate to health benefits. 'YES'",
        )
        chat_is_topic_appropriate.add_message(
            Role.USER,
            get_instruction(
                "Julia Sets",
                "In this image, there is a very detailed and complicated looking wave pattern or fractal type pattern on a purple background. The image also includes numbers and arrows pointing to different parts of the wave formation.",
            ),
        )
        chat_is_topic_appropriate.add_message(
            Role.ASSISTANT,
            "The fractal type patterns in the image may represent Julia Sets. 'YES'",
        )
        chat_is_topic_appropriate.add_message(
            Role.USER,
            get_instruction(topic, image_content),
        )
        is_topic_appropriate_response: str = self.session.generate_completion(
            chat_is_topic_appropriate, "orca2"
        )
        return is_topic_appropriate_response

    @classmethod
    def few_shot_generateBlackboardCaption(
        cls, topic: str, image_title: str, llm: str
    ) -> str:
        def get_instruction(l_topic: str, l_image_title: str) -> str:
            return (
                f"Compose a concise, instructive chalkboard caption for the topic '{l_topic}', "
                f"to complement an illustrative image titled '{l_image_title}'. "
                "Use Rich Text Formatting to enhance readability and emphasis. "
                "The caption should be brief yet comprehensive, encapsulating essential ideas and "
                "concepts pivotal for grasping the fundamentals of the topic."
            )

        chat_chalkboard_caption: Chat = Chat(
            "You are a helpful AI assistant. You comply with the users' requests by responding factually and concisely."
        )

        # First example conversation
        chat_chalkboard_caption.add_message(
            Role.USER,
            get_instruction(
                "Exploring the Mandelbrot Set: A Journey into Fractal Geometry",
                "The image shows a fractal pattern which is likely related to the Mandelbrot set.",
            ),
        )
        chat_chalkboard_caption.add_message(
            Role.ASSISTANT,
            """Sure!
'''chalkboard_caption
<u><b>Mandelbrot Set Overview</b></u>

<color=#808080><i>Definition:</i></color>
- Complex numbers: <color=#00BFFF>Real</color> and <color=purple>Imaginary</color> parts.

<color=#808080><i>Formula:</i></color>
- <color=green>z<sub>n+1</sub> = z<sub>n</sub>^2 + c</color>: Heart of fractal iterations.

<color=#808080><i>Fractal Nature:</i></color>
- Infinite complexity, <color=orange>self-similar</color> patterns at every scale.

<color=#808080><i>Visual Beauty:</i></color>
- Colors indicate <color=red>divergence speed</color>: A spectrum in chaos.'''""",
        )

        # Second example conversation
        chat_chalkboard_caption.add_message(
            Role.USER,
            get_instruction(
                "The Incredible Journey: Human Evolution",
                "The image shows an Infochart about the timeline of human evolution.",
            ),
        )
        chat_chalkboard_caption.add_message(
            Role.ASSISTANT,
            """Sure!
'''chalkboard_caption
<u><b>Human Evolution: An Incredible Journey</b></u>

<color=#008000><i>Key Milestones:</i></color>
- <color=#800080>Australopithecus:</color> The first step in bipedalism.
- <color=#FFA500>Homo habilis:</color> Early tool usage begins.
- <color=#1E90FF>Homo erectus:</color> Migration out of Africa.
- <color=#FF4500>Neanderthals:</color> Adaptation to colder climates.
- <color=#2E8B57>Modern Humans:</color> Development of complex societies.

<color=#808080><i>Evolutionary Significance:</i></color>
- Physical and cognitive changes over millennia.
- Adaptation to diverse environments and climates.

<color=#808080><i>Current Understanding:</i></color>
- Ongoing research and discoveries continuously reshape our understanding of human evolution.'''""",
        )

        # User's dynamic request
        chat_chalkboard_caption.add_message(
            Role.USER, get_instruction(topic, image_title)
        )

        # Generate blackboard text
        blackboard_text: str = cls.session.generate_completion(
            chat_chalkboard_caption,
            llm,
            "Sure!\n'''chalkboard_caption\n",
        )
        return blackboard_text

    # @classmethod
    # def few_shot_topicToSearch(
    #     self, topic: str, image_content: str, llm: str
    # ):
    #     chat_topic_to_search: Chat = Chat(
    #         "You are a topic to search-term converter. Respond with a well known term directly related to an visualization of the user given topic."
    #     )

    #     def get_instruction(search_topic: str):
    #         return f"Please provide a google searchterm for finding a good visualization of: '{search_topic}'"

    #     def get_response(image_search_term: str):
    #         return f"Sure! You should be able to find appropriate visualizations by searching for: '{image_search_term}'"

    #     chat_topic_to_search.add_message(
    #         Role.USER, get_instruction("Random Walks")
    #     )
    #     chat_topic_to_search.add_message(
    #         Role.ASSISTANT,
    #         get_response("Random Walk Monte Carlo Visualization"),
    #     )
    #     chat_topic_to_search.add_message(
    #         Role.USER, get_instruction("Natural Deduction")
    #     )
    #     chat_topic_to_search.add_message(
    #         Role.ASSISTANT,
    #         get_response("Natural Deduction Rule Diagram"),
    #     )
    #     chat_topic_to_search.add_message(
    #         Role.USER, get_instruction("Cell Division")
    #     )
    #     chat_topic_to_search.add_message(
    #         Role.ASSISTANT,
    #         get_response("Mitosis and Meiosis Stages Diagram"),
    #     )
    #     chat_topic_to_search.add_message(
    #         Role.USER, get_instruction("Cognitive Behavioral Therapy")
    #     )
    #     chat_topic_to_search.add_message(
    #         Role.ASSISTANT, get_response("CBT Techniques Infographic")
    #     )
    #     chat_topic_to_search.add_message(
    #         Role.USER, get_instruction("Electoral Systems")
    #     )
    #     chat_topic_to_search.add_message(
    #         Role.ASSISTANT,
    #         get_response("Comparative Electoral Systems Chart"),
    #     )
    #     chat_topic_to_search.add_message(
    #         Role.USER, get_instruction("Renewable Energy Sources")
    #     )
    #     chat_topic_to_search.add_message(
    #         Role.ASSISTANT,
    #         get_response("Solar and Wind Energy Infographic"),
    #     )
    #     chat_topic_to_search.add_message(
    #         Role.USER, get_instruction("Human Evolution")
    #     )
    #     chat_topic_to_search.add_message(
    #         Role.ASSISTANT, get_response("Hominid Evolutionary Tree")
    #     )
    #     chat_topic_to_search.add_message(
    #         Role.USER, get_instruction("Cellular Automata")
    #     )
    #     chat_topic_to_search.add_message(
    #         Role.ASSISTANT,
    #         get_response("Conways Game of Life"),
    #     )
    #     chat_topic_to_search.add_message(Role.USER, get_instruction(topic))
    #     search_term = self.session.generate_completion(
    #         chat_topic_to_search,
    #         self.llm,
    #         "Sure! You should be able to find appropriate visualizations by searching for: '",
    #     )
    #     search_term = search_term.replace(
    #         "Sure! You should be able to find appropriate visualizations by searching for: '",
    #         "",
    #     ).replace("'", "")
