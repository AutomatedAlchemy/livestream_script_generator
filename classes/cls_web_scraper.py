import base64
import logging
import random
from io import BytesIO
from typing import Callable, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import ddg
from PIL import Image


def validate_base64_image(base64_string: str) -> bool:
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        image.verify()
        if image.format == "GIF" or min(image.size) < 50:
            return False
        return True
    except Exception:
        return False


class WebScraper:
    def __init__(self):
        self.user_agents: List[str] = self._load_user_agents()
        self.urls: List[str] = []

    def _load_user_agents(self) -> List[str]:
        return [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        ]

    def duckduckgo_search(self, keyword: str, num_results: int) -> List[str]:
        try:
            results = ddg(keyword, max_results=num_results)
            return [r['href'] for r in results]
        except Exception as e:
            logging.error(f"Error with DuckDuckGo Search: {e}")
            return []

    def fetch_url_content(self, url: str) -> Optional[str]:
        headers = {"User-Agent": random.choice(self.user_agents)}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logging.error(f"Error fetching URL content: {e}")
            return None

    @staticmethod
    def _extract_text(html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()
        text_elements = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])
        return " ".join(element.get_text().strip() for element in text_elements if element.get_text())

    def scrape_web(self, keyword: str, num_sites: int = 3) -> List[str]:
        self.urls = self.duckduckgo_search(keyword, num_sites)
        contents: List[str] = []
        for url in self.urls:
            page_content = self.fetch_url_content(url)
            if page_content:
                extracted_text = self._extract_text(page_content)
                contents.append(extracted_text)
        return contents

    def fetch_image(self, url: str) -> Optional[bytes]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logging.error(f"Error fetching image: {e}")
            return None

    def get_next_image_as_base64(self, process_image: Callable[[str], bool]) -> Optional[str]:
        for url in self.urls:
            page_content = self.fetch_url_content(url)
            if page_content:
                soup = BeautifulSoup(page_content, "html.parser")
                image_tags = self._sort_images_by_quality(soup.find_all("img", src=True))

                for tag in image_tags:
                    image_url = self._get_high_quality_image_url(tag, url)
                    image_content = self.fetch_image(image_url)

                    if image_content:
                        base64_encoded = base64.b64encode(image_content).decode()
                        if validate_base64_image(base64_encoded) and process_image(base64_encoded):
                            return base64_encoded
        return None

    @staticmethod
    def _sort_images_by_quality(image_tags: List[BeautifulSoup]) -> List[BeautifulSoup]:
        def image_quality_score(tag: BeautifulSoup) -> int:
            srcset = tag.get("srcset")
            if srcset:
                resolutions = srcset.split(",")
                max_resolution_part = resolutions[-1].strip().split(" ")
                if len(max_resolution_part) >= 2:
                    resolution_factor = max_resolution_part[-2]
                    try:
                        return int(resolution_factor.replace("w", "").replace("x", ""))
                    except ValueError:
                        return 0
            return 0

        return sorted(image_tags, key=image_quality_score, reverse=True)

    @staticmethod
    def _get_high_quality_image_url(tag: BeautifulSoup, base_url: str) -> str:
        if tag.get("srcset"):
            srcset_urls = tag["srcset"].split(",")
            high_res_url = srcset_urls[-1].split(" ")[0].strip()
            return urljoin(base_url, high_res_url)

        return urljoin(base_url, tag.get("data-src") or tag["src"])


# Example usage:
# scraper = WebScraper()
# texts = scraper.scrape_web("example keyword")
# image_base64 = scraper.get_next_image_as_base64(process_image=lambda img: True)
