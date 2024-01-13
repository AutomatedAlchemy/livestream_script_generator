import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, Iterator, List, Optional

import requests
from jinja2 import Template

from interface.cls_chat import Chat, Role

# Configurations
BASE_URL = "http://localhost:11434/api"
TIMEOUT = 960  # Timeout for API requests in seconds
OLLAMA_CONTAINER_NAME = "ollama"  # Name of the Ollama Docker container
OLLAMA_START_COMMAND = [
    "docker",
    "run",
    "-d",
    "--gpus=all",
    "-v",
    "ollama:/root/.ollama",
    "-p",
    "11434:11434",
    "--name",
    OLLAMA_CONTAINER_NAME,
    "ollama/ollama",
]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class OllamaClient(metaclass=SingletonMeta):
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self._ensure_container_running()
        self.cache_file = "./cache/ollama_cache.json"
        self.cache = self._load_cache()

    def _ensure_container_running(self):
        """Ensure that the Ollama Docker container is running."""
        if self._check_container_exists():
            if not self._check_container_status():
                logger.info("Restarting the existing Ollama Docker container...")
                self._restart_container()
        else:
            logger.info("Starting a new Ollama Docker container...")
            self._start_container()

    def _check_container_status(self):
        """Check if the Ollama Docker container is running."""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "inspect",
                    '--format="{{ .State.Running }}"',
                    OLLAMA_CONTAINER_NAME,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip().strip('"') == "true"
        except subprocess.CalledProcessError:
            return False

    def _check_container_exists(self):
        """Check if a Docker container with the Ollama name exists."""
        result = subprocess.run(
            ["docker", "ps", "-a", "-q", "--filter", f"name={OLLAMA_CONTAINER_NAME}"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() != ""

    def _restart_container(self):
        """Restart the existing Ollama Docker container."""
        subprocess.run(["docker", "restart", OLLAMA_CONTAINER_NAME], check=True)

    def _start_container(self):
        """Start the Ollama Docker container."""
        try:
            subprocess.run(OLLAMA_START_COMMAND, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                "Error starting the Ollama Docker container. Please check the Docker setup."
            )
            raise

    def _download_model(self, model_name: str):
        """Download the specified model if not available."""
        logger.info(f"Checking if model '{model_name}' is available...")
        if not self._is_model_available(model_name):
            logger.info(f"Model '{model_name}' not found. Downloading...")
            subprocess.run(
                ["docker", "exec", OLLAMA_CONTAINER_NAME, "ollama", "pull", model_name],
                check=True,
            )
            logger.info(f"Model '{model_name}' downloaded.")

    def _is_model_available(self, model_name: str) -> bool:
        """Check if a specified model is available in the Ollama container."""
        result = subprocess.run(
            ["docker", "exec", OLLAMA_CONTAINER_NAME, "ollama", "list"],
            capture_output=True,
            text=True,
        )
        return model_name in result.stdout

    def _load_cache(self):
        """Load cache from a file."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        if not os.path.exists(self.cache_file):
            return {}  # Return an empty dictionary if file not found

        with open(self.cache_file, "r") as json_file:
            try:
                return json.load(json_file)  # Load and return cache data
            except json.JSONDecodeError:
                return {}  # Return an empty dictionary if JSON is invalid

    def _get_cached_completion(self, model: str, temperature: str, prompt: str) -> str:
        """Retrieve cached completion if available."""
        cache_key = f"{model}:{temperature}:{prompt}"
        return self.cache.get(cache_key)

    def _update_cache(self, model: str, temperature: str, prompt: str, completion: str):
        """Update the cache with new completion."""
        cache_key = f"{model}:{temperature}:{prompt}"
        self.cache[cache_key] = completion
        with open(self.cache_file, "w") as json_file:
            json.dump(self.cache, json_file, indent=4)

    def _send_request(
        self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None
    ) -> requests.Response:
        """Send an HTTP request to the given endpoint."""
        url = f"{self.base_url}/{endpoint}"

        try:
            if method == "GET":
                response = requests.get(url, timeout=TIMEOUT)
            elif method == "POST":
                if endpoint == "generate":
                    start_time = time.time()  # Record the start time
                    request_info = f"Sending request to model: {data['model']}..."
                    print(
                        request_info, end=""
                    )  # Print first part of the message without newline
                response = requests.post(url, json=data, timeout=TIMEOUT)
                if endpoint == "generate":
                    end_time = time.time()  # Record the end time
                    duration = end_time - start_time  # Calculate the duration
                    print(
                        f" Took {duration:.2f} seconds"
                    )  # Print second part of the message
            elif method == "DELETE":
                response = requests.delete(url, json=data, timeout=TIMEOUT)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            return response
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            raise

    def _get_template(self, model: str):
        data = {"name": model}
        response = self._send_request("POST", "show", data).json()
        if "error" in response:
            self._download_model(model)
            response = self._send_request("POST", "show", data).json()
        template_str: str = response["template"]
        template_str = template_str.replace(".Prompt", "prompt").replace(
            ".System", "system"
        )
        if (
            template_str
            == "{{ if system }}System: {{ system }}{{ end }}\nUser: {{ prompt }}\nAssistant:"
        ):
            return "{% if system %}System: {{ system }}{% endif %}\nUser: {{ prompt }}\nAssistant:"
        if (
            template_str
            == "{{- if system }}\n<|system|>\n{{ system }}\n</s>\n{{- end }}\n<|user|>\n{{ prompt }}\n</s>\n<|assistant|>\n"
        ):
            return "{% if system %}\n\n{{ system }}\n</s>\n{% endif %}\n\n{{ prompt }}\n</s>\n\n"
        if (
            template_str == "{{- if system }}\n### System:\n{{ system }}\n{{- end }}\n\n### User:\n{{ prompt }}\n\n### Response:\n"
        ):
            return "{% if system %}\n### System:\n{{ system }}\n{% endif %}\n\n### User:\n{{ prompt }}\n\n### Response:\n"
        if (template_str == "{{- if system }}\n<|im_start|>system {{ system }}<|im_end|>\n{{- end }}\n<|im_start|>user\n{{ prompt }}<|im_end|>\n<|im_start|>assistant\n"):
            return "{% if system %}\nsystem {{ system }}\n{% endif %}\nuser\n{{ prompt }}\nassistant"

        return template_str

    def generate_completion(
        self,
        prompt,
        model: str,
        start_response_with: str = "",
        instruction: str = "Your are an accurate and creative expert assistant. Converse in a proactive and precise manner.",
        temperature: float = 0.8,
        # stream: bool = False,
        **kwargs,
    ) -> str:
        template_str = self._get_template(model)
        # Remove the redundant addition of start_response_with
        if isinstance(prompt, Chat):
            prompt_str = prompt.to_jinja2(template_str)
        else:
            template = Template(template_str)
            context = {"system": instruction, "prompt": prompt}
            prompt_str = template.render(context)

        prompt_str += start_response_with

        # Check cache first
        cached_completion = self._get_cached_completion(model, temperature, prompt_str)
        if cached_completion:
            print(f"Cache hit! For: {model}")
            return start_response_with + cached_completion
        # If not cached, generate completion

        data = {
            "model": model,
            "prompt": prompt_str,
            "temperature": temperature,
            "raw": bool(instruction),
            # "stream": stream,
            **kwargs,
        }
        response = self._send_request("POST", "generate", data)

        # Revised approach to handle streaming JSON responses
        full_response = ""
        for line in response.text.strip().split("\n"):
            try:
                json_obj = json.loads(line)
                full_response += json_obj.get("response", "")
                if json_obj.get("done", False):
                    break
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding error in line: {line} - {e}")

        # Update cache
        self._update_cache(model, temperature, prompt_str, full_response)

        return start_response_with + full_response
    
    
    def str_to_list(self, list_str: str) -> List[str]:
        chat = Chat()
        chat.add_message(
            Role.SYSTEM,
            "Split the provided users text into a json array of context dependent strings.",
        )
        chat.add_message(
            Role.USER,
            "DeepMind's Gemini is a groundbreaking AI model designed for multimodality, capable of reasoning across various formats like text, images, video, audio, and code. It represents a significant advancement in AI, offering enhanced problem-solving and knowledge application abilities. Gemini notably outperforms human experts in Massive Multitask Language Understanding (MMLU) and sets new standards in benchmarks across text and coding, as well as multimodal benchmarks involving images, video, and audio. It comes in three versions - Ultra, Pro, and Nano - each tailored for different levels of complexity and tasks. Gemini's unique feature is its ability to transform any input into any output, demonstrating versatility in code generation and other applications. Additionally, DeepMind emphasizes responsible development and deployment of Gemini, incorporating safety measures and striving for inclusiveness. Gemini is accessible through platforms like Bard and Google AI Studio.",
        )
        chat.add_message(
            Role.ASSISTANT,
            '\'\'\'json["DeepMind\'s Gemini: A Groundbreaking Multimodal AI Model", "Capabilities in Reasoning Across Text, Images, Video, Audio, and Code", "Gemini\'s Superior Performance in MMLU and Multimodal Benchmarks", "Variants of Gemini: Ultra, Pro, and Nano for Different Complexity Levels", "Focus on Responsible Development and Deployment with Accessibility Features"]\'\'\'',
        )
        chat.add_message(
            Role.USER,
            "The following guide shows how our script needs to be implemented:\n1.Split the users input into characters.\n2.Use the characters to enable colouring.\n3.Print the users colour.",
        )
        chat.add_message(
            Role.ASSISTANT,
            "'''json[\"Split the users input into characters.\", \"Use the characters to enable colouring.\", \"Print the users colour.\"]'''",
        )
        chat.add_message(
            Role.USER,
            "The German Bundestag is set to legalize cannabis by April 1, 2024, a move delayed from the original January 1, 2024 date. This legislation, considered a major shift in Germany's drug policy, allows for controlled use of cannabis, including a distribution limit of 25 grams and the right to grow up to three plants. It aims to improve safety and reduce the burden on police and judiciary by moving away from the unregulated black market. Additionally, the website discusses a proposed hospital reform for quality improvement in German healthcare. Both initiatives reflect significant changes in public policy and health management in Germany.",
        )
        chat.add_message(
            Role.ASSISTANT,
            '\'\'\'json["Legalization of Cannabis in Germany", "Delay in Cannabis Legislation", "Controlled Use of Cannabis", "Proposed Hospital Reform in Germany", "Changes in Public Policy and Health Management"]\'\'\'',
        )
        chat.add_message(
            Role.USER,
            "Sure!\n1.What are the greatest economic challenges of 2024?\n2.What can we learn from 2023?\n3.Will climate change impact the price of fur coats?\n4.Should vegans be sentenced to carrotts?",
        )
        chat.add_message(
            Role.ASSISTANT,
            '\'\'\'json["What are the greatest economic challenges of 2024?", "What can we learn from 2023?", "Will climate change impact the price of fur coats?", "Should vegans be sentenced to carrots?"]\'\'\'',
        )

        chat.add_message(Role.USER, list_str)
        json_response = self.generate_completion(chat, "orca2", "'''json[\"")
        extracted_object_str = json_response.split("'''json")[1].split("'''")[0]
        return json.loads(extracted_object_str)


ollama_client = OllamaClient()
