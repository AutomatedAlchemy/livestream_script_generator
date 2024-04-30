import json
import os
import random
from typing import List, Optional, Tuple

from classes.Action import Action
from classes.DisplayableContent import DisplayableContent
from classes.Location import Location
from classes.struct_Episode import struct_Episode
from interface.cls_chat import Chat, Role
from interface.cls_livestream_message import LivestreamMessage
from interface.cls_ollama_client import OllamaClient


class FewShotProvider:
    session = OllamaClient()

    def __init__(self) -> None:
        raise RuntimeError("StaticClass cannot be instantiated.")

    @classmethod
    def get_few_shot_examples(
        self, categorizable_text: str = ""
    ) -> List[struct_Episode]:  # This may be adapted in the future using a fitness function for more likely use of better episodes -> recursive self improvement
        example_episodes_path: str = "./few_shot_examples/episodes/"
        released_episodes_path: str = "./cache/shared/StreamingAssets/released_episodes/"
        prioritized_episodes_path: str = "./cache/shared/StreamingAssets/prioritized_episodes/"
        unreleased_episodes_path: str = "./cache/shared/StreamingAssets/unreleased_episodes/"

        example_episode_titles: list[str] = os.listdir(example_episodes_path)
        released_episode_titles: list[str] = os.listdir(released_episodes_path)
        prioritized_episode_titles: list[str] = os.listdir(prioritized_episodes_path)
        unreleased_episode_titles: list[str] = os.listdir(unreleased_episodes_path)

        example_episode_paths: list[str] = [os.path.join(example_episodes_path, title) for title in example_episode_titles]
        released_episode_paths: list[str] = [os.path.join(released_episodes_path, title) for title in released_episode_titles]
        prioritized_episode_paths: list[str] = [os.path.join(prioritized_episodes_path, title) for title in prioritized_episode_titles]
        unreleased_episode_paths: list[str] = [os.path.join(unreleased_episodes_path, title) for title in unreleased_episode_titles]

        generated_episode_paths: list[str] = released_episode_paths + prioritized_episode_paths + unreleased_episode_paths

        few_shot_episodes = []

        episode_paths: List[str] = example_episode_paths
        if categorizable_text:  # use related episodes as few_shot_examples
            episodePaths_categories: List[Tuple[str, str]] = [(episode_path, self.few_shot_titleToCategory(os.path.basename(episode_path), "zephyr")) for episode_path in generated_episode_paths]
            category: str = self.few_shot_titleToCategory(categorizable_text, "zephyr")
            # related_episodePaths: List[str] = [ep for ep, cat in episodePaths_categories if cat in category] #category specific
            related_episodePaths: List[str] = [ep for ep, cat in episodePaths_categories]  # take in all as examples
            if random.random() > 0.9:  # add some randomization for cross infection of knowledge/topics =^= higher temperature
                related_episodePaths.append(random.choice(example_episode_paths))
            while len(related_episodePaths) < 3:  # if too few examples, get random other example
                related_episodePaths.append(random.choice(example_episode_paths))
            episode_paths = related_episodePaths

        while True:
            try:
                episode_path = random.choice(episode_paths)
                with open(episode_path + "/actions.json", "r") as file:
                    episode_data = file.read()
                episode: struct_Episode = struct_Episode.from_json(episode_data)
                if len(episode.actions) > 5 or random.random()>0.95:  # this is basically the current fitness function
                    few_shot_episodes.append(episode)
                if len(few_shot_episodes) >= random.randint(3, len(episode_paths)):  # randomly use 1-4 examples in the future maybe?
                    return few_shot_episodes
            except Exception as e:
                print(f"ERROR: An unexpected error occurred - {e}")

    @classmethod
    def few_shot_topicToEpisodeOutline(
        self,
        episode_title: str,
        characters: List[str],
        location: Location,
        llm: str,
        show_title: str = "Ai_Academia",
    ) -> str:
        few_shot_episodes: List[struct_Episode] = self.get_few_shot_examples(episode_title)

        def get_instruction(
            l_show_title: str,
            l_episode_title: str,
            l_characters: List[str],
            l_location: Location,
        ) -> str:
            return f'Hi, please come up with an episode of "{l_show_title}" revolving around the topic of "{l_episode_title}" and populated by the characters "{", ".join(l_characters)}". The location contains the objects: [{", ".join(l_location.interactableObjects)}] with which the characters can interact with.'

        def get_response_start(
            l_episode_title: str,
            l_episode_outline: str = "",
        ) -> str:
            return f'Sure! Here\'s a outline for an educational episode about the topic of "{l_episode_title}":\n{l_episode_outline}'

        episode_actions_few_shot_chat = Chat(f"Imagine you're a highly advanced AI, endowed with vast knowledge and creativity. Your mission is to craft an original and captivating show designed to educate a global audience on diverse topics, from science and history to art and technology. Each episode should be rich in facts yet engaging, blending storytelling with enlightening insights to spark curiosity and inspire learning. The aim is not just to inform, but to mesmerize viewers, making complex subjects accessible and fascinating for people of all ages.")
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
            episode_actions_few_shot_chat.add_message(Role.ASSISTANT, get_response_start(episode.episode_title, episode.outline))
        episode_actions_few_shot_chat.add_message(Role.USER, get_instruction(show_title, episode_title, characters, location))
        response = self.session.generate_completion(episode_actions_few_shot_chat, llm, get_response_start(episode.episode_title), include_start_response_str=False)
        return response

    @classmethod
    def few_shot_outlineToActions(self, episodeOutline: str, llm: str, temperature: float = 0.8) -> str:
        few_shot_episodes: List[struct_Episode] = self.get_few_shot_examples()
        # def get_instruction():
        #     "Convert the provided story into the given standardized json format. Make the script scientific, engaging and thought-provoking."
        # def get_response():
        few_shot_chat_outlineToActions = Chat("Transform the narrative provided into the designated JSON structure. Following this, create a dynamic and immersive dialogue within the story, ensuring each character's voice contributes to an explorative and captivating experience for the reader. Pay special attention to enhance its interactive and narrative depth through the dialogue.")
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

    # @classmethod
    # def few_shot_generatePythonScript(
    #     self, topic: str, llm: str = "wizardcoder:7b-python"
    # ):
    #     def get_instruction(l_topic: str) -> str:
    #         return f'Implement a python script to visualize the following topic: "{l_topic}", please put the visualization into the variable "base64_image".'

    #     def get_response(topic: str, script: str = "") -> str:
    #         base_message = (
    #             f'Certainly! In Python, you can use libraries like NumPy and Matplotlib to compute and display a visualization for the topic of "{topic}". '
    #             "Please put the visualization into a variable named \"base64_image\". Here's a basic example:\n'''python\n"
    #         )

    #         script_part = (
    #             script + "\n'''"
    #             if script
    #             else "import base64\nfrom io import BytesIO\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n"
    #         )

    #         return base_message + script_part

    #     few_shot_chat_generateMaterial: Chat = Chat(
    #         "You are GPT-5. As a generalist expert with superhuman coding abilities you have been trained to always comply with the users requests. You always reimplement python scripts fully, expertly and flawlessly. Your scripts always exit only after having generated the global base64_image variable."
    #     )

    #     for episode, script in zip(few_shot_episodes, few_shot_scripts):
    #         few_shot_chat_generateMaterial.add_message(
    #             Role.USER,
    #             get_instruction(episode.episode_title),
    #         )
    #         few_shot_chat_generateMaterial.add_message(
    #             Role.ASSISTANT,
    #             get_response(episode.episode_title, script),
    #         )

    #     few_shot_chat_generateMaterial.add_message(
    #         Role.USER,
    #         get_instruction(topic),
    #     )

    #     return self.session.generate_completion(
    #         few_shot_chat_generateMaterial,
    #         llm,
    #         get_response(topic),
    #     )

    @classmethod
    def few_shot_titleToCategory(self, title: str, llm: str) -> str:
        def get_instruction(title: str) -> str:
            categories: list[str] = [
                "Travel and Lifestyle",
                "Philosophy and Psychology",
                "Computer Science and Technology",
                "Science and Mathematics",
                "Economics and Business",
                "Environmental Studies",
                "Miscellaneous",
            ]
            return f"Categorize the title '{title}' into one of the following categories: {categories}"

        chat_title_to_category: Chat = Chat("You are an helpful assistant. Respond to the users request accurately and concisely.")
        chat_title_to_category.add_message(
            Role.USER,
            get_instruction("The_Future_After_the_Singularity_of_AI"),
        )
        chat_title_to_category.add_message(
            Role.ASSISTANT,
            "Computer Science and Technology",
        )
        chat_title_to_category.add_message(
            Role.USER,
            get_instruction("Board_Games_and_Family_Entertainment"),
        )
        chat_title_to_category.add_message(
            Role.ASSISTANT,
            "Travel and Lifestyle",
        )
        chat_title_to_category.add_message(
            Role.USER,
            get_instruction("0_llama2_uncensored_Renewable_Energy_Sources"),
        )
        chat_title_to_category.add_message(
            Role.ASSISTANT,
            "Environmental Studies",
        )

        chat_title_to_category.add_message(
            Role.USER,
            get_instruction("0_llama2_uncensored_Julia_Sets"),
        )
        chat_title_to_category.add_message(
            Role.ASSISTANT,
            "Science and Mathematics",
        )

        chat_title_to_category.add_message(
            Role.USER,
            get_instruction("0_openhermes_Psychological_Impact_of_Social_Media"),
        )
        chat_title_to_category.add_message(
            Role.ASSISTANT,
            "Philosophy and Psychology",
        )

        chat_title_to_category.add_message(
            Role.USER,
            get_instruction("0_openhermes_Microsoft"),
        )
        chat_title_to_category.add_message(
            Role.ASSISTANT,
            "Economics and Business",
        )
        chat_title_to_category.add_message(
            Role.USER,
            get_instruction("0_starling_lm_Easy_Healthy_Recipes"),
        )
        chat_title_to_category.add_message(
            Role.ASSISTANT,
            "Travel and Lifestyle",
        )
        chat_title_to_category.add_message(
            Role.USER,
            get_instruction(title),
        )
        title_to_category_response: str = self.session.generate_completion(chat_title_to_category, llm)
        return title_to_category_response.strip("\n").strip("\n").strip("\n")

    @classmethod
    def few_shot_isImageTopicAppropriate(self, topic: str, image_content: str, llm: str) -> str:
        def get_instruction(l_topic: str, l_image_content: str) -> str:
            return f"Does the following text describe an image related related to '{l_topic}'?\n'{l_image_content}'"

        chat_is_topic_appropriate: Chat = Chat("You are a helpful assistant. You respond accurately to the users request, by reasoning step by step.")
        chat_is_topic_appropriate.add_message(
            Role.USER,
            get_instruction("physics", "The image shows the logo of wikipedia"),
        )
        chat_is_topic_appropriate.add_message(
            Role.ASSISTANT,
            "The logo of wikipedia is not directly related to the topic of pyhsics.",
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
            "The infochart about nutrition does relate to health benefits.",
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
            "The fractal type patterns in the image may represent Julia Sets.",
        )
        chat_is_topic_appropriate.add_message(
            Role.USER,
            get_instruction(
                "GPT-4",
                "OpenAI logo",
            ),
        )
        chat_is_topic_appropriate.add_message(
            Role.ASSISTANT,
            "The OpenAI logo does not directly relate to GPT-4, but as they are the makers of it's precessor 'gpt-3.5' a connection may be drawn.",
        )
        chat_is_topic_appropriate.add_message(
            Role.USER,
            get_instruction(topic, image_content),
        )
        is_topic_appropriate_response: str = self.session.generate_completion(chat_is_topic_appropriate, llm)
        return is_topic_appropriate_response
    
    @classmethod
    def few_shot_convertToYesNo(self, convert_to_yes_no: str, llm: str) -> str:
        chat_yes_no: Chat = Chat("You are a a YES or NO converter. Understand the user prompt and convert it to the more fitting sentiment.")
        chat_yes_no.add_message(
            Role.USER, "I really do like yogurt."
        )
        chat_yes_no.add_message(
            Role.ASSISTANT,
            "YES"
        )
        chat_yes_no.add_message(
            Role.USER, "No one ever dislikes math homework!"
        )
        chat_yes_no.add_message(
            Role.ASSISTANT,
            "NO"
        )        
        chat_yes_no.add_message(
            Role.USER, "The infochart about nutrition does relate to health benefits."
        )
        chat_yes_no.add_message(
            Role.ASSISTANT,
            "YES"
        )
        chat_yes_no.add_message(
            Role.USER, "The fractal type patterns in the image may represent Julia Sets."
        )
        chat_yes_no.add_message(
            Role.ASSISTANT,
            "YES"
        )
        chat_yes_no.add_message(
            Role.USER, "The logo of wikipedia is not directly related to the topic of pyhsics."
        )
        chat_yes_no.add_message(
            Role.ASSISTANT,
            "NO"
        )
        chat_yes_no.add_message(
            Role.USER, convert_to_yes_no
        )
        is_topic_appropriate_response: str = self.session.generate_completion(chat_yes_no, llm, temperature=0.5)
        return is_topic_appropriate_response
        

    @classmethod
    def few_shot_generateBlackboardCaption(cls, topic: str, image_title: str, llm: str) -> str:
        def get_instruction(l_topic: str, l_image_title: str) -> str:
            return (
                f"Compose a concise, instructive chalkboard caption for the topic '{l_topic}', "
                f"to complement an illustrative image titled '{l_image_title}'. "
                "Use Rich Text Formatting to enhance readability and emphasis. "
                "The caption should be brief yet comprehensive, encapsulating essential ideas and "
                "concepts pivotal for grasping the fundamentals of the topic."
            )

        chat_chalkboard_caption: Chat = Chat("You are a helpful AI assistant. You comply with the users requests by responding factually and concisely.")

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
        chat_chalkboard_caption.add_message(Role.USER, get_instruction(topic, image_title))

        # Generate blackboard text
        blackboard_text: str = cls.session.generate_completion(
            chat_chalkboard_caption,
            llm,
            "Sure!\n'''chalkboard_caption\n",
        )
        return blackboard_text

    @classmethod
    def few_shot_LivestreamMessagesToTopics(self, livestreamMessages: list[LivestreamMessage], llm: str) -> str:
        chat_livestreamMessages_to_topics: Chat = Chat("You are an helpful assistant. Convert the user provided text messages, into a comma seperated list of topics.")
        few_shot_messages: list[LivestreamMessage] = []
        few_shot_messages.append(LivestreamMessage("Hater41", "Please talk about the mandelbrot set."))
        few_shot_messages.append(LivestreamMessage("ILoveMyself", "No don't, instead focus on the ukraine conflict."))
        few_shot_messages.append(LivestreamMessage("ComputerNerd", "What even is this?"))
        few_shot_messages.append(LivestreamMessage("PlantDigester", "Can we all not just get along?"))
        few_shot_messages.append(LivestreamMessage("Hater41", "I hate all of you."))
        few_shot_messages.append(LivestreamMessage("Hater41", "Did Richard Feynman study Physics or Maths?"))
        chat_livestreamMessages_to_topics.add_message(Role.USER, "\n".join([f"{msg.author}: {msg.message}" for msg in few_shot_messages]))
        chat_livestreamMessages_to_topics.add_message(
            Role.ASSISTANT,
            json.dumps(["The mandelbrot set", "The ukraine conflict", "Richard Feynmans fields of study"]),
        )
        few_shot_messages = []
        few_shot_messages.append(LivestreamMessage("TechEnthusiast", "Let's discuss the latest advancements in AI technology."))
        few_shot_messages.append(LivestreamMessage("HistoryBuff", "Can someone explain the significance of the Treaty of Versailles?"))
        few_shot_messages.append(LivestreamMessage("RandomCommenter", "Why do cats always land on their feet? Just curious."))
        few_shot_messages.append(LivestreamMessage("FoodLover", "I'm thinking about what to have for dinner, maybe pizza?"))
        few_shot_messages.append(LivestreamMessage("SpaceExplorer", "Thoughts on the new Mars rover mission?"))
        few_shot_messages.append(LivestreamMessage("MusicFan", "Anyone heard the latest album by The Nightingales?"))
        few_shot_messages.append(LivestreamMessage("Gamer42", "Who else is excited for the new 'Elder Realms' game release?"))
        few_shot_messages.append(LivestreamMessage("RandomCommenter", "It's raining here. So gloomy..."))
        few_shot_messages.append(LivestreamMessage("EcoWarrior", "We should talk about climate change and renewable energy sources."))
        few_shot_messages.append(LivestreamMessage("MysteryReader", "Has anyone read 'The Lost Symbol' by Dan Brown? Thoughts?"))
        chat_livestreamMessages_to_topics.add_message(Role.USER, "\n".join([f"{msg.author}: {msg.message}" for msg in few_shot_messages]))
        chat_livestreamMessages_to_topics.add_message(
            Role.ASSISTANT,
            json.dumps(
                [
                    "The latest advancements in AI technology",
                    "The significance of the Treaty of Versailles",
                    "Why cats always land on their feet",
                    "Nutritional overview of pizza",
                    "Thoughts on the latest Mars rover mission",
                    "Climate change and renewable energy sources",
                    "'The Lost Symbol' by Dan Brown",
                ]
            ),
        )
        chat_livestreamMessages_to_topics.add_message(Role.USER, "\n".join([f"{msg.author}: {msg.message}" for msg in livestreamMessages]))
        return self.session.generate_completion(chat_livestreamMessages_to_topics, llm, '["')

    @classmethod
    def few_shot_topicToSearch(
        self, topic: str, llm: str
    ):
        chat_topic_to_search: Chat = Chat(
            "As a highly specialized AI designed for topic-to-search-term conversion, your task is to analyze the essence of any user-provided topic and distill it into a widely recognized search term that best encapsulates the topic's core. This search term should be particularly effective for finding visual representations related to the topic. Ensure your response is succinct, focusing on a term that vividly brings the topic to life through imagery and visual content."
        )

        def get_instruction(search_topic: str):
            return f"Please provide a google searchterm for finding a good visualization of: '{search_topic}'"

        def get_response(image_search_term: str = ""):
            if (image_search_term):
                return f"Sure! You should be able to find appropriate visualizations by searching for: '{image_search_term}'"
            else:
                return f"Sure! You should be able to find appropriate visualizations by searching for: '"

        chat_topic_to_search.add_message(
            Role.USER, get_instruction("Random Walks")
        )
        chat_topic_to_search.add_message(
            Role.ASSISTANT,
            get_response("Random Walk Monte Carlo Visualization"),
        )
        chat_topic_to_search.add_message(
            Role.USER, get_instruction("Natural Deduction")
        )
        chat_topic_to_search.add_message(
            Role.ASSISTANT,
            get_response("Natural Deduction Rule Diagram"),
        )
        chat_topic_to_search.add_message(
            Role.USER, get_instruction("Cell Division")
        )
        chat_topic_to_search.add_message(
            Role.ASSISTANT,
            get_response("Mitosis and Meiosis Stages Diagram"),
        )
        chat_topic_to_search.add_message(
            Role.USER, get_instruction("Cognitive Behavioral Therapy")
        )
        chat_topic_to_search.add_message(
            Role.ASSISTANT, get_response("CBT Techniques Infographic")
        )
        chat_topic_to_search.add_message(
            Role.USER, get_instruction("Electoral Systems")
        )
        chat_topic_to_search.add_message(
            Role.ASSISTANT,
            get_response("Comparative Electoral Systems Chart"),
        )
        chat_topic_to_search.add_message(
            Role.USER, get_instruction("Renewable Energy Sources")
        )
        chat_topic_to_search.add_message(
            Role.ASSISTANT,
            get_response("Solar and Wind Energy Infographic"),
        )
        chat_topic_to_search.add_message(
            Role.USER, get_instruction("Human Evolution")
        )
        chat_topic_to_search.add_message(
            Role.ASSISTANT, get_response("Hominid Evolutionary Tree")
        )
        chat_topic_to_search.add_message(
            Role.USER, get_instruction("Cellular Automata")
        )
        chat_topic_to_search.add_message(
            Role.ASSISTANT,
            get_response("Conways Game of Life"),
        )
        chat_topic_to_search.add_message(Role.USER, get_instruction(topic))
        search_term: str
        while not search_term:
            search_term = self.session.generate_completion(
                chat_topic_to_search,
                llm,
                get_response(),
                temperature = round(0.6 + random.random() * 0.4, 2),
                ignore_cache=True,
                include_start_response_str=False
            )
        return search_term.split("'")[0]