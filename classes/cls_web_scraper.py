import base64
import logging
import random
import time
from io import BytesIO
from typing import Callable, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import ddg
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_base64_image(base64_string: str) -> bool:
    """Validates if a base64 encoded string is a valid non-GIF image with a minimum size of 250px by 250px."""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        image.verify()
        # Update minimum size check to 250px by 250px
        return image.format != "GIF" and min(image.size) >= 250
    except Exception as e:
        logging.error(f"Image validation failed: {e}")
        return False

def correct_url_scheme(url: str) -> str:
    """Ensures the URL has a valid https scheme if it lacks one."""
    parsed = urlparse(url)
    if not parsed.scheme:
        return f"https:{url}" if url.startswith("//") else f"https://{url}"
    return url

class WebScraper:
    """Web scraper for fetching high quality images from search results."""
    def __init__(self, search_keyword: str):
        self.search_keyword = search_keyword
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        ]
        self.urls: List[str] = []
        self.processed_urls:set = set()
        self.url_ignore_list:list = []
        self.max_retries = 100
        self.retry_sleep = 1

    def fetch_url_content(self, url: str) -> Optional[str]:
        corrected_url = correct_url_scheme(url)
        if corrected_url in self.url_ignore_list or corrected_url in self.processed_urls:
            return None
        self.processed_urls.add(corrected_url)
        headers = {"User-Agent": random.choice(self.user_agents)}
        try:
            response = requests.get(corrected_url, headers=headers, timeout=5)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            self.url_ignore_list.append(corrected_url)
            logging.error(f"Failed to fetch URL content: {e} for URL: {corrected_url}")
            return None

    def fetch_image(self, url: str) -> Optional[bytes]:
        corrected_url = correct_url_scheme(url)
        if corrected_url in self.url_ignore_list or corrected_url in self.processed_urls:
            return None
        self.processed_urls.add(corrected_url)
        headers = {"User-Agent": random.choice(self.user_agents)}
        try:
            response = requests.get(corrected_url, headers=headers, timeout=5)
            response.raise_for_status()
            logging.info(f"Downloaded image from: {corrected_url}")
            return response.content
        except requests.RequestException as e:
            self.url_ignore_list.append(corrected_url)
            logging.error(f"Failed to fetch image: {e} for URL: {corrected_url}")
            return None

    def duckduckgo_search(self, keyword: str) -> List[str]:
        try:
            results = ddg(keyword, max_results=10)
            return [result["href"] for result in results]
        except Exception as e:
            logging.error(f"DuckDuckGo search failed: {e}")
            return []

    def process_page_for_image(self, page_content: str, process_image: Callable[[str], bool]) -> Optional[str]:
        soup = BeautifulSoup(page_content, "html.parser")
        image_tags = soup.find_all("img", src=True)
        random.shuffle(image_tags)
        for tag in image_tags:
            image_url = self.get_high_quality_image_url(tag, soup.base.get('href') if soup.base else "")
            if image_url and (not image_url.endswith(".svg")):
                image_content = self.fetch_image(image_url)
                if image_content:
                    base64_encoded = base64.b64encode(image_content).decode()
                    if validate_base64_image(base64_encoded) and process_image(base64_encoded):
                        return base64_encoded
        return None

    def get_high_quality_image_url(self, tag, base_url: str) -> Optional[str]:
        # Prioritize high-resolution versions by checking additional attributes
        src = tag.get("data-srcset") or tag.get("data-src") or tag.get("srcset") or tag.get("src")
        # Attempt to select the highest resolution available in srcset if present
        if src and ',' in src:
            src = max((piece.strip().split(' ')[0] for piece in src.split(',')), key=lambda x: int(x.split('w')[-1]), default=src)
        corrected_url = correct_url_scheme(urljoin(base_url, src))
        return corrected_url

    def get_images_as_base64(self, process_image: Callable[[str], bool]) -> Optional[str]:
        retry_count = 0
        while retry_count < self.max_retries:
            if not self.urls:
                self.urls = self.duckduckgo_search(self.search_keyword)
                self.urls = [url for url in self.urls if url not in self.url_ignore_list]
                random.shuffle(self.urls)
            for url in self.urls:
                page_content = self.fetch_url_content(url)
                if page_content:
                    base64_image = self.process_page_for_image(page_content, process_image)
                    if base64_image:
                        return base64_image
            self.urls = []
            retry_count += 1
            time.sleep(self.retry_sleep)
        logging.error("Max retries reached without finding a suitable image.")
        return None

# Note: The script assumes the existence of a functioning ddg() function for DuckDuckGo searches,
# which needs to be replaced or mocked if not available in the current environment.
