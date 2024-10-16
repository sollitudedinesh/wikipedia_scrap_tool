import requests
from bs4 import BeautifulSoup

def scrape_wikipedia_page(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return content
    else:
        raise Exception(f"Failed to fetch page: {response.status_code}")
