import aiohttp
import asyncio
from bs4 import BeautifulSoup
from markdownify import markdownify as md

URLS = [
    'https://www.nhs.uk/mental-health/conditions/phobias/overview/',
    'https://www.nhs.uk/mental-health/conditions/phobias/symptoms/',
    'https://www.nhs.uk/mental-health/conditions/phobias/causes/',
    'https://www.nhs.uk/mental-health/conditions/phobias/treatment/',
    'https://www.nhs.uk/mental-health/conditions/phobias/self-help/'
]


async def fetch_url(session, url):
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


async def fetch_all_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(fetch_url(session, url))

        results = await asyncio.gather(*tasks)
        return results


def get_urls_async(urls):
    return asyncio.run(fetch_all_urls(urls))


def main():
    scrapped_pages = get_urls_async(URLS)
    for i, pages in enumerate(scrapped_pages):
        soup = BeautifulSoup(pages, "lxml")
        content = soup.find(attrs={'id': 'maincontent'})
        markdown = md(content.decode().replace('\n', ''))
        with open(f'../data/nhs/text{i}.md', 'w') as f:
            f.write(markdown)
            f.flush()


if __name__ == '__main__':
    main()


