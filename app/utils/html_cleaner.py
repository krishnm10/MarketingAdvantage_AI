# app/utils/html_cleaner.py

from bs4 import BeautifulSoup

REMOVE_TAGS = ["script", "style", "noscript", "footer", "nav"]


def clean_html(raw_html: str) -> str:
    """
    Hybrid HTML Cleaner:
    - Removes script/style/nav/footer/ads
    - Extracts clean readable text
    """

    if not raw_html:
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove noisy tags
    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()

    for node in soup.find_all(text=True):
        if node.parent.name in REMOVE_TAGS:
            node.extract()

    clean = soup.get_text(separator=" ")

    # Remove duplicate spaces
    clean = " ".join(clean.split())

    return clean
