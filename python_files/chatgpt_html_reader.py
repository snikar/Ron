"""
ChatGPT HTML Reader (Phase 1)
-----------------------------
Reads ChatGPT's exported HTML file and extracts raw conversation text.
"""

from bs4 import BeautifulSoup
from importers.html_cleaner import clean_html_text


class ChatGPTHTMLReader:
    """
    Responsible for loading ChatGPT export HTML and pulling raw strings
    from message bubbles.
    """

    def __init__(self):
        pass

    def read_html(self, file_bytes: bytes) -> list:
        """
        Receives raw HTML bytes.
        Returns a list of raw text blocks extracted from ChatGPT messages.
        """

        soup = BeautifulSoup(file_bytes, "html.parser")

        # ChatGPT exports vary slightly; target the main message containers
        message_divs = soup.find_all("div")

        texts = []
        for div in message_divs:
            # Extract text but ignore nav, sidebar, buttons, etc.
            raw = div.get_text(strip=True)
            if not raw:
                continue

            cleaned = clean_html_text(raw)

            # Avoid junk from CSS and scripts
            if len(cleaned) < 5:
                continue
            if "ChatGPT" in cleaned and len(cleaned) < 20:
                continue

            texts.append(cleaned)

        return texts
