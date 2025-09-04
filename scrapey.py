import time
import re
import requests
import unicodedata
import hashlib
from bs4 import BeautifulSoup
from openai import OpenAI
from ddgs import DDGS
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

# Connect to LM Studio's local API server (still used for query generation)
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
ddgs = DDGS()

def smart_deduplicate(results):
    """
    Deduplicate results by their 'body' content.
    Keeps results with no body to avoid losing URLs.
    """
    seen_hashes = set()
    deduped = []
    for r in results:
        content = r.get("body")
        if content:
            h = hashlib.md5(content.encode("utf-8")).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped.append(r)
        else:
            # Keep the URL if no content
            deduped.append(r)
    return deduped


def count_words_in_texts(texts):
    """
    Count total number of words in a list of text blocks.
    """
    total_words = sum(len(t.split()) for t in texts)
    return total_words


def sanitize_text_for_llm(text: str) -> str:
    """
    Clean text to make it safe for AI datasets (e.g., Alpaca).
    Removes control characters, emojis, unsupported symbols, and excessive whitespace.
    """
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)

    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

    # Remove emojis/pictographs
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002500-\U00002BEF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)

    # Keep only printable characters
    text = ''.join(c for c in text if c.isprintable())

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def clean_generated_queries(raw_queries, n=100):
    """
    Remove junk lines (instructions, numbering, empty, etc.)
    Keep only plausible queries.
    """
    queries = []
    for q in raw_queries:
        q = q.strip()
        # Skip if empty
        if not q:
            continue
        # Skip meta-lines
        if q.lower().startswith("here are"):
            continue
        if q.lower().startswith("alternative"):
            continue
        # Skip if it's just numbers or too short
        if q.isdigit() or len(q.split()) < 2:
            continue
        # Strip leading numbers / punctuation (e.g. "1. query")
        q = re.sub(r'^[0-9]+[).:\- ]+', '', q).strip()
        queries.append(q)
    # Deduplicate while preserving order
    return list(dict.fromkeys(queries))[:n]


def is_mostly_english(text: str, threshold: float = 0.9) -> bool:
    """
    True if the text is mostly English-looking (ASCII letters/spaces).
    threshold = fraction of chars that must be ASCII letters/spaces.
    """
    if not text:
        return False
    english_chars = sum(1 for c in text if c.isascii() and (c.isalpha() or c.isspace()))
    ratio = english_chars / max(1, len(text))
    return ratio >= threshold


def generate_multiple_queries(user_prompt, n=50):
    """
    Generate multiple search queries from a single user prompt.
    Returns a list of queries.
    """
    try:
        response = client.chat.completions.create(
            model="google/gemma-3-1b",
            messages=[
                {"role": "system", "content": "You are an assistant that converts a user's prompt into multiple concise search queries."},
                {"role": "user", "content": f"User prompt: {user_prompt}\n\nGenerate {n} alternate and unique search queries (MAX 20 words each), with no punctuation and try to keep them simple. You are allowed to generate different queries that will bring up different search information, as long as the searches they bring up will be viable for answering the user's question. Output only the queries, one per line with NO explanation or any other text, ONLY the queries. DO NOT NUMBER THE QUERIES."}
            ],
            temperature=0.2
        )
        queries = [q.strip() for q in response.choices[0].message.content.split("\n") if q.strip()]
        queries = clean_generated_queries(queries, n=n)
        return queries
    except Exception as e:
        print(f"Error generating multiple queries: {e}")
        return [user_prompt]  # fallback to single query


def sanitize_query(query: str) -> str:
    query = re.sub(r'[^A-Za-z0-9 ]+', '', query)
    return ' '.join(query.split()[:12])


def enhanced_search(query: str, max_results: int = 1000, log_raw: bool = True):
    """
    Search text and news results, deduplicate by URL.
    """
    all_results = []
    seen_urls = set()
    try:
        # Pre-warm
        _ = list(ddgs.text("test", max_results=10))
        _ = list(ddgs.news("test", max_results=10))
        time.sleep(1)

        # Fetch results
        text_results = list(ddgs.text(query, max_results=max_results))
        news_results = list(ddgs.news(query, max_results=max_results))

        if log_raw:
            print(f"[RAW SEARCH] Query: '{query}' | Text: {len(text_results)}, News: {len(news_results)}")

        for r in text_results + news_results:
            if "href" in r and r["href"] not in seen_urls:
                all_results.append(r)
                seen_urls.add(r["href"])
        return all_results[:max_results]

    except Exception as e:
        print(f"Error searching internet: {e}")
        return all_results


def fetch_full_page_text(url, max_pages=5, query=None):
    """
    Fetch the main text content of a web page, follow pagination if possible.
    Keeps mostly-English, substantive paragraphs only.
    Fallback: Use DDGS snippets if page fails.
    """
    text_blocks = []
    try:
        for _ in range(max_pages):
            resp = requests.get(url, timeout=10)
            # Let requests choose encoding unless missing
            if resp.encoding is None:
                resp.encoding = resp.apparent_encoding
            html = resp.text

            soup = BeautifulSoup(html, 'html.parser')

            # Remove non-visible/script content
            for tag in soup(['script', 'style', 'noscript']):
                tag.decompose()

            # Collect paragraphs/articles/list-items, filter to English + substantive
            blocks = []
            for el in soup.find_all(['p', 'article', 'li']):
                t = el.get_text(separator=' ', strip=True)
                if t and len(t) >= 50 and is_mostly_english(t):
                    blocks.append(t)

            # De-dup lines while preserving order
            if blocks:
                dedup = list(dict.fromkeys(blocks))
                text_blocks.append("\n".join(dedup))

            # Find "next" page
            next_link = (
                soup.find('a', attrs={'rel': 'next'}) or
                soup.find('a', string=re.compile(r'\b(next|older)\b', re.I))
            )
            if not next_link or 'href' not in next_link.attrs:
                break
            url = urljoin(url, next_link['href'])

    except Exception as e:
        print(f"Error fetching {url}: {e}")
        if query:
            # DDGS fallback for this query
            print("Using DDGS fallback...")
            try:
                snippets = list(ddgs.text(query, max_results=10))
                eng_snippets = []
                for s in snippets:
                    body = sanitize_text_for_llm(s.get('body', ''))
                    if body and is_mostly_english(body):
                        eng_snippets.append(body)
                if eng_snippets:
                    text_blocks.append("\n".join(eng_snippets))
            except Exception as e2:
                print(f"DDGS fallback failed: {e2}")

    return "\n".join(t for t in text_blocks if t)


def save_search_results_to_file(results, filename="search_results.txt", fetch_full_content=True, chunk_size=100):
    """
    Save sanitized full page content for AI dataset.
    Splits into chunks of `chunk_size` words, separated by blank lines.
    Uses multi-threading for faster scraping.
    """
    def fetch_and_sanitize(r):
        if "href" in r:
            full = fetch_full_page_text(r["href"], query=r.get('body', None))
            cleaned = sanitize_text_for_llm(full)
            # Keep only mostly-English, reasonably long pages
            if cleaned and len(cleaned) > 300 and is_mostly_english(cleaned, threshold=0.85):
                return cleaned
        return ""

    def split_into_chunks(text, chunk_size=100):
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    with ThreadPoolExecutor(max_workers=10) as executor:
        sanitized_texts = list(executor.map(fetch_and_sanitize, results))

    sanitized_texts = [t for t in sanitized_texts if t]  # remove empty
    with open(filename, "w", encoding="utf-8") as f:
        for text in sanitized_texts:
            chunks = split_into_chunks(text, chunk_size=chunk_size)
            for chunk in chunks:
                f.write(chunk + "\n\n")

    print(f"Saved {len(sanitized_texts)} search results to '{filename}' (chunked into {chunk_size}-word blocks)")


if __name__ == "__main__":
    # Step 0: Read user prompt
    with open("query.txt", "r", encoding="utf-8") as f:
        user_prompt = f.read().strip()

    # Step 1: Generate multiple search queries
    query_list = generate_multiple_queries(user_prompt, n=100)

    # Step 1b: Run searches
    all_results = []
    for q in query_list:
        sanitized_q = sanitize_query(q)
        results = enhanced_search(sanitized_q, max_results=1000)
        all_results.extend(results)

    # Step 2: Deduplicate by URL
    unique_results = {r["href"]: r for r in all_results if "href" in r}.values()
    search_results = list(unique_results)

    # Step 2b: Smart deduplication by content
    search_results = smart_deduplicate(search_results)

    # Step 3: Save sanitized content
    CHUNK_SIZE = 100  # change here if you want bigger/smaller chunks
    save_search_results_to_file(search_results, fetch_full_content=True, chunk_size=CHUNK_SIZE)

    # Step 4: Count words in saved dataset
    with open("search_results.txt", "r", encoding="utf-8") as f:
        saved_texts = f.read().split("\n")
    total_words = count_words_in_texts(saved_texts)
    print(f"Total words in dataset: {total_words}")



