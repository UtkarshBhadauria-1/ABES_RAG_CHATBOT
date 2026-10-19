import json
import time
import hashlib
import urllib.robotparser

from pathlib import Path
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree

import requests


# ---------------------------------
# 1. Configuration
# ---------------------------------

BASE_URL = "https://www.abes.ac.in/"
SITEMAP_URL = "https://www.abes.ac.in/sitemap.xml"

ALLOWED_HOST = "www.abes.ac.in"

MAX_PAGES = 5
REQUEST_DELAY = 10

USER_AGENT = "ABESResearchBot/1.0 (educational project)"

RAW_DIR = Path("data/raw/html")
MANIFEST_FILE = Path("data/manifests/pages.jsonl")

RAW_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------
# 2. HTTP session
# ---------------------------------

session = requests.Session()

session.headers.update({
    "User-Agent": USER_AGENT
})


# ---------------------------------
# 3. Retrieve robots.txt
# ---------------------------------

def load_robots():

    robots_url = BASE_URL + "robots.txt"

    response = session.get(
        robots_url,
        timeout=20
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Could not retrieve robots.txt: "
            f"HTTP {response.status_code}. "
            "Stop and investigate access."
        )

    robots = urllib.robotparser.RobotFileParser()

    robots.set_url(robots_url)
    robots.parse(response.text.splitlines())

    return robots


# ---------------------------------
# 4. Retrieve sitemap URLs
# ---------------------------------

def get_sitemap_urls():

    response = session.get(
        SITEMAP_URL,
        timeout=20
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Could not retrieve sitemap: "
            f"HTTP {response.status_code}"
        )

    root = ElementTree.fromstring(response.content)

    namespace = {
        "sm": "http://www.sitemaps.org/schemas/sitemap/0.9"
    }

    urls = []

    for element in root.findall("sm:url/sm:loc", namespace):

        if not element.text:
            continue

        url, _ = urldefrag(element.text.strip())

        parsed = urlparse(url)

        if (
            parsed.scheme == "https"
            and parsed.hostname == ALLOWED_HOST
        ):
            urls.append(url)

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(urls))


# ---------------------------------
# 5. Save HTML and metadata
# ---------------------------------

def save_page(url, html, title, status_code):

    filename = hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest() + ".html"

    filepath = RAW_DIR / filename

    filepath.write_text(
        html,
        encoding="utf-8"
    )

    record = {
        "url": url,
        "title": title,
        "status_code": status_code,
        "html_file": str(filepath),
        "content_type": "text/html"
    }

    with MANIFEST_FILE.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


# ---------------------------------
# 6. Crawl sitemap pages
# ---------------------------------

def crawl_website():

    robots = load_robots()
    urls = get_sitemap_urls()

    print("URLs discovered in sitemap:", len(urls))

    successful = 0

    for url in urls:

        if successful >= MAX_PAGES:
            break

        if not robots.can_fetch(USER_AGENT, url):

            print("Disallowed by robots.txt:", url)
            continue

        try:

            response = session.get(
                url,
                timeout=20
            )

            if response.status_code != 200:

                print(
                    "Request failed:",
                    response.status_code,
                    url
                )

                # Do not retry or attempt to evade
                # server-side access restrictions.
                continue

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if "text/html" not in content_type:

                print("Skipping non-HTML:", url)
                continue

            html = response.text

            # Extract the page title.
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(
                html,
                "lxml"
            )

            title = (
                soup.title.get_text(
                    " ",
                    strip=True
                )
                if soup.title
                else "Untitled"
            )

            save_page(
                url,
                html,
                title,
                response.status_code
            )

            successful += 1

            print(
                f"[{successful}/{MAX_PAGES}] "
                f"Saved: {title}"
            )

            time.sleep(REQUEST_DELAY)

        except requests.RequestException as error:

            print("Request error:", url, error)

    print("\nCrawl finished.")
    print("Successfully downloaded:", successful)
    print("Sitemap URLs discovered:", len(urls))


# ---------------------------------
# 7. Run crawler
# ---------------------------------

if __name__ == "__main__":

    crawl_website()