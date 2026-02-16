import requests
from bs4 import BeautifulSoup 
from fastapi import HTTPException

def fetch_clean_text(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text[:12000]
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to fetch URL")