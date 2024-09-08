import aiohttp
import asyncio
from bs4 import BeautifulSoup
import os

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


def extract_filename_from_url(url):
    return url.rstrip('/').split('/')[-1]


def main():
    # Ensure the directory exists
    output_dir = os.path.join('data_collection', 'data', 'nhs')
    os.makedirs(output_dir, exist_ok=True)

    scrapped_pages = get_urls_async(URLS)
    for i, page in enumerate(scrapped_pages):
        if page:
            soup = BeautifulSoup(page, "html.parser")
            content = soup.find(attrs={'id': 'maincontent'})
            text_content = content.get_text(separator='\n').strip()
            
            filename = extract_filename_from_url(URLS[i])
            filepath = os.path.join(output_dir, f'{filename}.txt')
            
            with open(filepath, 'w') as f:
                f.write(text_content)
                f.flush()


if __name__ == '__main__':
    main()
