import asyncio
from pathlib import Path
import re
from playwright.async_api import async_playwright

async def crawl_and_save(start_url: str, max_links: int = 3, output_dir: str = "output"):
    # Prepare output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def sanitize_filename(url: str) -> str:
        # Remove scheme, replace non-alphanumeric with underscores
        name = re.sub(r'https?://', '', url)
        name = re.sub(r'[^0-9a-zA-Z]+', '_', name)
        return name.strip('_') + ".html"

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Crawl the initial page
        await page.goto(start_url, wait_until="networkidle", timeout=30000)
        html = await page.content()
        filename = sanitize_filename(start_url)
        (out_dir / filename).write_text(html, encoding="utf-8")
        print(f"Saved initial page to {out_dir / filename}")

        # Extract and crawl links
        links = await page.eval_on_selector_all(
            "a[href]", "els => els.map(e => e.href)"
        )
        print(f"Found {len(links)} links. Saving up to {max_links} of them.")

        for idx, link in enumerate(links[:max_links], start=1):
            try:
                await page.goto(link, wait_until="networkidle", timeout=30000)
                sub_html = await page.content()
                fname = sanitize_filename(link)
                (out_dir / fname).write_text(sub_html, encoding="utf-8")
                print(f"{idx}. Saved {link} to {out_dir / fname}")
            except Exception as e:
                print(f"{idx}. Failed to save {link}: {e}")

        await browser.close()

if __name__ == "__main__":
    import sys
    start = sys.argv[1] if len(sys.argv) > 1 else "https://qstommyshu.github.io/"
    max_links = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    asyncio.run(crawl_and_save(start, max_links))
