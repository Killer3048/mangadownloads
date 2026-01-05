import asyncio
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import aiohttp
import yaml
from PIL import Image, ImageFile
from playwright.async_api import async_playwright


SCRIPT_DIR = Path(__file__).resolve().parent


async def ensure_naver_login(playwright, config: dict) -> 'BrowserContext':
    """
    Ensure user is logged into Naver using a persistent browser profile.
    
    On first run (no profile): Opens browser, user logs in and closes browser manually.
    On subsequent runs: Loads existing profile with all session data in headless mode.
    
    Returns the browser context with active Naver session.
    """
    user_data_dir = config.get("naver_user_data_dir", "naver_profile")
    headless_mode = config.get("headless", True)
    
    # Build full path for user data directory
    user_data_path = SCRIPT_DIR / user_data_dir
    profile_exists = user_data_path.exists() and any(user_data_path.iterdir()) if user_data_path.exists() else False
    
    print(f"[NAVER] Using persistent browser profile: {user_data_path}")
    
    # Prepare proxy settings if needed
    use_browser_proxy = config.get("use_proxy_for_browser", False)
    proxy_url = config.get("proxy", "") if use_browser_proxy else ""
    
    launch_options = {}
    if proxy_url:
        proxy_match = re.match(r'http://([^:]+):([^@]+)@([^:]+):(\d+)', proxy_url)
        if proxy_match:
            proxy_user, proxy_pass, proxy_host, proxy_port = proxy_match.groups()
            print(f"[NAVER] Using proxy for browser: {proxy_host}:{proxy_port}")
            launch_options["proxy"] = {
                "server": f"http://{proxy_host}:{proxy_port}",
                "username": proxy_user,
                "password": proxy_pass
            }
        else:
            launch_options["proxy"] = {"server": proxy_url}
    
    # If profile exists, use it in headless mode
    if profile_exists:
        print("[NAVER] ✓ Profile found! Loading saved session...")
        context = await playwright.chromium.launch_persistent_context(
            user_data_dir=str(user_data_path),
            headless=headless_mode,
            **launch_options
        )
        return context
    
    # First run - need manual login
    print("\n" + "=" * 60)
    print("[NAVER] ⚠ FIRST RUN - Manual login required!")
    print("=" * 60)
    print("[NAVER] Browser will open now.")
    print("[NAVER] Please:")
    print("[NAVER]   1. Log in to your Naver account")
    print("[NAVER]   2. Browse around if you want (everything is saved)")
    print("[NAVER]   3. Close the browser window when done (click X)")
    print("[NAVER] The script will continue after you close the browser.")
    print("=" * 60 + "\n")
    
    # Launch visible browser for first-time login
    context = await playwright.chromium.launch_persistent_context(
        user_data_dir=str(user_data_path),
        headless=False,  # Always visible for first-time login
        **launch_options
    )
    
    # Get or create first page
    if context.pages:
        page = context.pages[0]
    else:
        page = await context.new_page()
    
    # Navigate to login page
    try:
        await page.goto("https://nid.naver.com/nidlogin.login", timeout=30000)
    except Exception as e:
        print(f"[NAVER] Error navigating to login page: {e}")
    
    # Wait for user to close the browser
    # When all pages are closed, context.pages will be empty
    print("[NAVER] Waiting for you to close the browser...")
    
    try:
        # Wait until context is closed (user closes browser window)
        while True:
            await asyncio.sleep(1)
            try:
                # Check if context is still alive by checking pages
                _ = context.pages
            except Exception:
                # Context disconnected = browser closed
                break
            
            # Also check if all pages are closed
            if len(context.pages) == 0:
                await asyncio.sleep(2)  # Give a moment in case new tab is opening
                if len(context.pages) == 0:
                    break
    except Exception:
        pass
    
    print("[NAVER] ✓ Browser closed! Session saved to:", user_data_path)
    print("[NAVER] Reopening browser in background mode...")
    
    # Now reopen in headless mode with saved profile
    context = await playwright.chromium.launch_persistent_context(
        user_data_dir=str(user_data_path),
        headless=headless_mode,
        **launch_options
    )
    
    return context

# 1) Disable Pillow's DecompressionBombError for large images
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_config():
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config.yaml: {e}")
        return {}
def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '', name).strip()


def sanitize_ascii(name):
    """Remove all non-Latin/non-ASCII characters from filename, keeping only safe characters."""
    # Keep only ASCII letters, digits, hyphens, underscores, and dots
    sanitized = re.sub(r'[^a-zA-Z0-9_.\-]', '', name)
    # Remove leading/trailing dots and collapse multiple dots into one
    sanitized = re.sub(r'\.+', '.', sanitized)  # Collapse multiple dots
    sanitized = sanitized.strip('.')  # Remove leading/trailing dots
    # If result is empty, use a fallback
    if not sanitized:
        sanitized = "chapter"
    return sanitized


def get_site_handler(url):
    """Determine which site handler to use based on URL."""
    if "comic.naver.com" in url:
        return "naver"
    elif "demonicscans.org" in url:
        return "demonic"
    elif "mangalib.me" in url:
        return "mangalib"
    elif "webtoons.com" in url:
        return "webtoons"
    return None


TOKI_BLOCKED_RE = re.compile(
    r"^https?://(?:newtoki\d+\.com/webtoon/\d+|manatoki\d+\.net/comic/\d+|booktoki\d+\.com/novel/\d+)"
)


def is_blocked_toki_url(url: str) -> bool:
    """Newtoki/Manatoki/Booktoki URLs are prohibited."""
    return bool(TOKI_BLOCKED_RE.match(url))


def parse_naver_url(url):
    """Extract titleId and chapter number from Naver Comic URL."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    title_id = params.get('titleId', [None])[0]
    chapter_no = params.get('no', [None])[0]
    return title_id, chapter_no


def parse_naver_episode(episode_text):
    """
    Parse episode text from Naver Comic.
    Examples:
        '197화' -> '197'
        '시즌2 01화' -> '2-01'
        '특별편' -> '특별편'
    """
    if not episode_text:
        return None
    
    episode_text = episode_text.strip()
    
    # Pattern for season format: "시즌2 01화" or "시즌2 1화"
    season_match = re.match(r'시즌(\d+)\s*(\d+)화?', episode_text)
    if season_match:
        season_num = season_match.group(1)
        ep_num = season_match.group(2).lstrip('0') or '0'  # Remove leading zeros
        return f"{season_num}-{ep_num.zfill(2)}"  # Format: 2-01
    
    # Pattern for simple episode: "197화"
    simple_match = re.match(r'(\d+)화?$', episode_text)
    if simple_match:
        return simple_match.group(1)
    
    # Fallback for special episodes like "특별편"
    # Sanitize for use as folder name
    return sanitize_filename(episode_text)


def parse_demonic_url(url):
    """Extract manga title and chapter from DemonicScans URL."""
    try:
        parts = url.strip('/').split('/')
        title_idx = parts.index('title') + 1
        chapter_idx = parts.index('chapter') + 1
        manga_title = parts[title_idx]
        chapter_num = parts[chapter_idx]
        return sanitize_filename(manga_title), chapter_num
    except:
        return None, None


def parse_mangalib_url(url):
    """
    Parse Mangalib URL to extract manga slug, volume and chapter.
    Example: https://mangalib.me/ru/80583--wo-zhen-bu-shi-xie-shen-zou-gou/read/v1/c167
    Returns: (slug, volume, chapter)
    """
    try:
        # Pattern: /ru/{slug}/read/v{volume}/c{chapter}
        match = re.search(r'/ru/([^/]+)/read/v(\d+)/c(\d+)', url)
        if match:
            slug = match.group(1)
            volume = match.group(2)
            chapter = match.group(3)
            return slug, volume, chapter
    except:
        pass
    return None, None, None


def parse_webtoons_url(url):
    """
    Parse Webtoons URL to extract title ID, title name, and episode number.
    Example: https://www.webtoons.com/en/fantasy/the-luckiest-mage/episode-1/viewer?title_no=9174&episode_no=1
    Returns: (title_no, title_slug, episode_no)
    """
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        title_no = params.get('title_no', [None])[0]
        episode_no = params.get('episode_no', [None])[0]
        
        # Extract title slug from path: /en/genre/title-slug/episode-X/viewer
        path_parts = parsed.path.strip('/').split('/')
        # Path format: lang/genre/title-slug/episode-X/viewer
        title_slug = None
        if len(path_parts) >= 3:
            title_slug = path_parts[2]  # title-slug
        
        return title_no, title_slug, episode_no
    except:
        pass
    return None, None, None


#
# NOTE: Newtoki/Manatoki/Booktoki support intentionally removed.
#


async def download_image(
    session: aiohttp.ClientSession,
    url: str,
    save_path: Path,
    referer: str,
    proxy: str,
    timeout_connect: int,
    timeout_read: int,
    retries: int,
) -> bool:
    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Referer": referer
            }
            
            timeout = aiohttp.ClientTimeout(sock_connect=timeout_connect, sock_read=timeout_read)
            
            # Parse proxy URL if provided
            proxy_url = None
            proxy_auth = None
            if proxy:
                # Parse: http://user:pass@host:port
                proxy_match = re.match(r'http://([^:]+):([^@]+)@(.+)', proxy)
                if proxy_match:
                    proxy_user = proxy_match.group(1)
                    proxy_pass = proxy_match.group(2)
                    proxy_host = proxy_match.group(3)
                    proxy_url = f"http://{proxy_host}"
                    proxy_auth = aiohttp.BasicAuth(proxy_user, proxy_pass)
                else:
                    proxy_url = proxy
            
            async with session.get(url, headers=headers, proxy=proxy_url, proxy_auth=proxy_auth, timeout=timeout) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                content = await response.read()
                
                if total_size > 0 and len(content) < total_size:
                     raise Exception(f"Incomplete download: {len(content)}/{total_size} bytes")

                save_path.parent.mkdir(parents=True, exist_ok=True)
                with save_path.open('wb') as f:
                    f.write(content)
                
                return True

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_err = e
        except Exception as e:
            last_err = e
            
        # Cleanup
        try:
            save_path.unlink(missing_ok=True)
        except Exception:
            pass

    if last_err is not None:
        print(f"Failed to download: {url} ({last_err})")
    return False


def create_merged_png(image_files: List[str], output_path: str) -> bool:
    """Merge a list of image files into one long vertical PNG."""
    if not image_files:
        return False

    files: List[Path] = [Path(p) for p in image_files]

    kept: List[tuple[Path, int, int]] = []
    total_height = 0
    max_width = 0

    for p in files:
        try:
            with Image.open(p) as im:
                width, height = im.size
            kept.append((p, width, height))
            total_height += height
            max_width = max(max_width, width)
        except Exception as e:
            print(f"Warning: could not read image {p}: {e}")

    if not kept:
        return False

    merged_img = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for p, _, height in kept:
        try:
            with Image.open(p) as im:
                merged_img.paste(im.convert("RGB"), (0, y_offset))
            y_offset += height
        except Exception as e:
            print(f"Warning: could not paste image {p}: {e}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged_img.save(output_path)
    return True


# ==================== BROWSER EXTRACTION ====================

async def scroll_page_fully(page):
    """Scroll page to trigger lazy loading of all images."""
    await page.evaluate("""
        async () => {
            await new Promise((resolve) => {
                let totalHeight = 0;
                let distance = 500;
                let timer = setInterval(() => {
                    let scrollHeight = document.body.scrollHeight;
                    window.scrollBy(0, distance);
                    totalHeight += distance;
                    if(totalHeight >= scrollHeight - window.innerHeight){
                        clearInterval(timer);
                        resolve();
                    }
                }, 100);
            });
        }
    """)
    await page.wait_for_timeout(2000)  # Wait for lazy load


async def extract_naver_images(page, url):
    """Extract images from Naver Comic page."""
    title_id, _ = parse_naver_url(url)  # chapter_no will be extracted from page
    if not title_id:
        print(f"Invalid Naver Comic URL: {url}")
        return None, None, []
    
    try:
        print(f"Loading Naver Comic page: {url}")
        await page.goto(url, timeout=120000, wait_until="domcontentloaded")
        await page.wait_for_selector(".wt_viewer", timeout=60000)
    except Exception as e:
        print(f"Failed to load Naver Comic page: {e}")
        return None, None, []
    
    # Get manga title from page
    try:
        manga_title = await page.evaluate("document.querySelector('#titleName_toolbar')?.innerText || ''")
        if not manga_title:
            manga_title = f"naver_{title_id}"
    except:
        manga_title = f"naver_{title_id}"
    
    manga_title = sanitize_filename(manga_title)
    
    # Get episode number from #subTitle_toolbar
    # Examples: "197화" -> "197", "시즌2 01화" -> "2-01"
    try:
        episode_text = await page.evaluate("document.querySelector('#subTitle_toolbar')?.innerText || ''")
        chapter_no = parse_naver_episode(episode_text)
        if not chapter_no:
            # Fallback to URL parameter
            _, chapter_no = parse_naver_url(url)
    except:
        _, chapter_no = parse_naver_url(url)
    
    # Scroll to load all lazy images
    await scroll_page_fully(page)
    
    # Extract image URLs
    image_urls = await page.evaluate("""
        () => {
            const images = document.querySelectorAll('.wt_viewer img');
            return Array.from(images).map(img => img.src).filter(src => src && src.startsWith('http'));
        }
    """)
    
    print(f"Found {len(image_urls)} images for {manga_title} Ch {chapter_no}")
    return manga_title, chapter_no, image_urls


async def extract_demonic_images(page, url):
    """Extract images from DemonicScans page."""
    manga_title, chapter_num = parse_demonic_url(url)
    if not manga_title or not chapter_num:
        print(f"Skipping invalid DemonicScans URL: {url}")
        return None, None, []
    
    current_url = url
    # Retry logic for DemonicScans (append /1)
    for attempt in range(2):
        try:
            print(f"Loading page: {current_url}")
            await page.goto(current_url, timeout=60000)
            await page.wait_for_selector("div.main-width.center-m", timeout=30000)
            break  # Success, content found
        except Exception as e:
            # Check if we should retry with /1
            is_demonic = "demonicscans.org" in current_url
            no_slash_one = not current_url.endswith("/1")
            
            if attempt == 0 and is_demonic and no_slash_one:
                print(f"Failed to load {current_url}. Retrying with /1 suffix...")
                current_url = f"{current_url}/1"
                continue
            else:
                print(f"Timeout waiting for content on {current_url}")
                return None, None, []
    
    # Scroll to load lazy images
    await scroll_page_fully(page)
    
    # Extract images
    images = await page.locator("div.main-width.center-m img.imgholder").all()
    image_urls = []
    for img in images:
        src = await img.get_attribute("src")
        if not src: src = await img.get_attribute("data-src")
        if src: image_urls.append(src)
    
    print(f"Found {len(image_urls)} images for {manga_title} Ch {chapter_num}")
    return manga_title, chapter_num, image_urls


async def extract_mangalib_images(page, url):
    """Extract images from Mangalib page with vertical mode switching."""
    slug, volume, chapter = parse_mangalib_url(url)
    if not slug or not chapter:
        print(f"Invalid Mangalib URL: {url}")
        return None, None, []
    
    try:
        print(f"Loading Mangalib page: {url}")
        await page.goto(url, timeout=120000, wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)  # Wait for page to stabilize
    except Exception as e:
        print(f"Failed to load Mangalib page: {e}")
        return None, None, []
    
    # Close cookie banner if present
    try:
        await page.evaluate("""
            () => {
                const okBtn = Array.from(document.querySelectorAll('button'))
                    .find(btn => btn.textContent.trim().toLowerCase() === 'ok');
                if (okBtn) okBtn.click();
            }
        """)
        await page.wait_for_timeout(500)
    except:
        pass
    
    # Switch to vertical mode by clicking the "Вертикальный" button
    try:
        print("Switching to vertical mode...")
        
        # Check if already in vertical mode
        current_mode = await page.evaluate("document.querySelector('[data-reader-mode]')?.dataset?.readerMode || ''")
        if current_mode == "vertical":
            print("Already in vertical mode")
        else:
            # Click on the gear icon (fa-gear) to open settings
            await page.evaluate("""
                () => {
                    // Find the gear icon by looking for fa-gear class
                    const gearIcon = document.querySelector('.fa-gear, [data-icon="gear"]');
                    if (gearIcon) {
                        // Click the parent element (the button container)
                        let parent = gearIcon.closest('.wr_mr, .wr_a8, button, div');
                        if (parent) parent.click();
                        else gearIcon.click();
                    }
                }
            """)
            await page.wait_for_timeout(1500)
            
            # Click "Вертикальный" button
            clicked = await page.evaluate("""
                () => {
                    const buttons = Array.from(document.querySelectorAll('button, div, span'));
                    const verticalBtn = buttons.find(el => {
                        const text = el.textContent.trim();
                        return text === 'Вертикальный' || text === 'Vertical';
                    });
                    if (verticalBtn) {
                        verticalBtn.click();
                        return true;
                    }
                    return false;
                }
            """)
            
            if clicked:
                print("Clicked 'Вертикальный' button")
                await page.wait_for_timeout(2000)  # Wait for mode to switch
                
                # Verify mode switched
                new_mode = await page.evaluate("document.querySelector('[data-reader-mode]')?.dataset?.readerMode || ''")
                if new_mode == "vertical":
                    print("Mode switched to vertical successfully")
                else:
                    print(f"Warning: Mode is still '{new_mode}'")
            else:
                print("Warning: Could not find 'Вертикальный' button")
    except Exception as e:
        print(f"Warning: Could not switch to vertical mode: {e}")
    
    # Get manga title from page
    try:
        manga_title = await page.evaluate("""
            () => {
                const titleLink = document.querySelector('a[href*="/ru/manga/"]');
                if (titleLink) {
                    const divs = titleLink.querySelectorAll('div');
                    if (divs.length >= 2) {
                        return divs[1].textContent.trim() || divs[0].textContent.trim();
                    }
                    return titleLink.textContent.trim();
                }
                return null;
            }
        """)
        if not manga_title:
            # Fallback: use slug
            manga_title = slug.split('--')[1] if '--' in slug else slug
    except:
        manga_title = slug.split('--')[1] if '--' in slug else slug
    
    manga_title = sanitize_filename(manga_title)
    
    # Format chapter number with volume
    chapter_num = f"v{volume}-c{chapter}"
    
    # Get total page count from the DOM
    total_pages = await page.evaluate("""
        () => {
            const pages = document.querySelectorAll('[data-page]');
            return pages.length;
        }
    """)
    print(f"Chapter has {total_pages} pages to load...")
    
    # Collect image URLs as we scroll - DOM is virtualized, images get removed!
    collected_urls = []
    
    # Scroll to each page container to trigger lazy loading and collect URL immediately
    for page_num in range(1, total_pages + 1):
        # Scroll the page container into view
        await page.evaluate(f"""
            () => {{
                const pageEl = document.querySelector('[data-page="{page_num}"]');
                if (pageEl) {{
                    pageEl.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                }}
            }}
        """)
        
        # Wait for image to appear and get its URL immediately
        image_url = None
        for _ in range(90):  # Max 45 seconds wait per image (large images)
            image_url = await page.evaluate(f"""
                () => {{
                    const pageEl = document.querySelector('[data-page="{page_num}"]');
                    if (pageEl) {{
                        const img = pageEl.querySelector('img.ace_mo');
                        if (img && img.src && img.src.startsWith('http')) {{
                            return img.src;
                        }}
                    }}
                    return null;
                }}
            """)
            if image_url:
                break
            await page.wait_for_timeout(500)
        
        if image_url:
            collected_urls.append(image_url)
            print(f"  Page {page_num}/{total_pages}: ✓")
        else:
            print(f"  Page {page_num}/{total_pages}: ✗ timeout")
    
    image_urls = collected_urls
    print(f"Found {len(image_urls)} images for {manga_title} Ch {chapter_num}")
    return manga_title, chapter_num, image_urls


async def extract_webtoons_images(page, url):
    """
    Extract images from Webtoons.com page.
    Images use class '_images' with actual URL in 'data-url' attribute.
    """
    title_no, title_slug, episode_no = parse_webtoons_url(url)
    if not title_no or not episode_no:
        print(f"Invalid Webtoons URL: {url}")
        return None, None, []
    
    try:
        print(f"Loading Webtoons page: {url}")
        await page.goto(url, timeout=120000, wait_until="domcontentloaded")
        await page.wait_for_selector("img._images", timeout=60000)
    except Exception as e:
        print(f"Failed to load Webtoons page: {e}")
        return None, None, []
    
    # Get manga title from page (need to extract clean title without episode info)
    try:
        manga_title = await page.evaluate("""
            () => {
                // Try og:title first: format is "Episode X | Title Name"
                const metaTitle = document.querySelector('meta[property="og:title"]')?.content;
                if (metaTitle) {
                    const parts = metaTitle.split('|');
                    if (parts.length >= 2) {
                        return parts[1].trim();
                    }
                }
                
                // Try the actual title link in the page header
                const titleLink = document.querySelector('a.subj_info')?.textContent;
                if (titleLink) {
                    return titleLink.trim();
                }
                
                // Try og:site_name as last resort for webtoon title
                const siteName = document.querySelector('meta[property="og:site_name"]')?.content;
                if (siteName && siteName !== 'WEBTOON') {
                    return siteName.trim();
                }
                
                return null;
            }
        """)
        if not manga_title:
            # Fallback: use slug from URL (e.g. "the-luckiest-mage" -> "The Luckiest Mage")
            manga_title = title_slug.replace('-', ' ').title() if title_slug else f"webtoons_{title_no}"
    except:
        manga_title = title_slug.replace('-', ' ').title() if title_slug else f"webtoons_{title_no}"
    
    manga_title = sanitize_filename(manga_title)
    
    # Scroll to load all lazy images
    print("Scrolling to load all images...")
    await page.evaluate("""
        async () => {
            let lastHeight = document.body.scrollHeight;
            while (true) {
                window.scrollTo(0, document.body.scrollHeight);
                await new Promise(r => setTimeout(r, 1500));
                let newHeight = document.body.scrollHeight;
                if (newHeight === lastHeight) break;
                lastHeight = newHeight;
            }
            // Scroll back to top to ensure all images are processed
            window.scrollTo(0, 0);
            await new Promise(r => setTimeout(r, 500));
        }
    """)
    
    # Wait a bit more for any remaining lazy loading
    await page.wait_for_timeout(2000)
    
    # Extract image URLs from data-url attribute
    image_urls = await page.evaluate("""
        () => {
            const images = document.querySelectorAll('img._images');
            return Array.from(images).map(img => {
                // Prefer data-url (actual image), fallback to src if already loaded
                const dataUrl = img.getAttribute('data-url');
                if (dataUrl && dataUrl.startsWith('http')) {
                    return dataUrl;
                }
                // Check if src is the actual image (not placeholder)
                const src = img.src;
                if (src && src.startsWith('http') && !src.includes('bg_transparency')) {
                    return src;
                }
                return dataUrl || src;
            }).filter(url => url && url.startsWith('http') && !url.includes('bg_transparency'));
        }
    """)
    
    print(f"Found {len(image_urls)} images for {manga_title} Episode {episode_no}")
    return manga_title, episode_no, image_urls


def guess_image_ext(img_url: str) -> str:
    try:
        ext = Path(urlparse(img_url).path).suffix.lower().lstrip(".")
    except Exception:
        ext = ""

    if ext == "jpeg":
        ext = "jpg"
    if ext in ("jpg", "png", "webp"):
        return ext
    return "jpg"


def build_chapter_paths(download_dir: str, manga_title: str, chapter_num: str) -> tuple[Path, Path, Path]:
    chapter_num_safe = sanitize_ascii(str(chapter_num))
    chapter_root = Path(download_dir) / manga_title / f"Chapter-{chapter_num_safe}"
    raw_frames_dir = chapter_root / "raw_frames"
    merged_path = chapter_root / "merged.png"
    return chapter_root, raw_frames_dir, merged_path


async def download_and_merge(
    *,
    download_dir: str,
    manga_title: str,
    chapter_num: str,
    image_urls: List[str],
    referer: str,
    config: Dict,
) -> Optional[str]:
    chapter_root, raw_frames_dir, merged_path = build_chapter_paths(download_dir, manga_title, chapter_num)
    raw_frames_dir.mkdir(parents=True, exist_ok=True)

    proxy = str(config.get("proxy", "") or "")
    retries = int(config.get("retries", 5) or 5)
    timeout_connect = int(config.get("timeout_connect", 5) or 5)
    timeout_read = int(config.get("timeout_read", 20) or 20)

    save_paths: List[Path] = []
    tasks: List[asyncio.Task[bool]] = []

    async with aiohttp.ClientSession() as session:
        for i, img_url in enumerate(image_urls):
            if not img_url or not img_url.startswith(("http://", "https://")):
                continue

            ext = guess_image_ext(img_url)
            save_path = raw_frames_dir / f"{i:03d}.{ext}"
            save_paths.append(save_path)
            tasks.append(
                asyncio.create_task(
                    download_image(
                        session,
                        img_url,
                        save_path,
                        referer,
                        proxy,
                        timeout_connect,
                        timeout_read,
                        retries,
                    )
                )
            )

        if not tasks:
            shutil.rmtree(chapter_root, ignore_errors=True)
            return None

        results = await asyncio.gather(*tasks)

    downloaded = [str(p) for p, ok in zip(save_paths, results) if ok]
    downloaded.sort()

    if not downloaded:
        shutil.rmtree(chapter_root, ignore_errors=True)
        return None

    ok = await asyncio.to_thread(create_merged_png, downloaded, str(merged_path))
    if not ok:
        shutil.rmtree(chapter_root, ignore_errors=True)
        return None

    shutil.rmtree(raw_frames_dir, ignore_errors=True)
    return str(chapter_root)


async def process_naver_chapter(url, context, config, semaphore) -> Optional[str]:
    """
    Process a single Naver chapter using existing persistent context.
    This preserves the session/cookies for Naver authentication.
    Returns the chapter root path.
    """
    async with semaphore:  # Limit concurrent tabs
        download_dir = config.get("download_dir", "Downloads")
        
        # Create a new page in the existing context (shares cookies/session)
        page = await context.new_page()
        
        try:
            manga_title, chapter_num, image_urls = await extract_naver_images(page, url)
            referer = "https://comic.naver.com/"
        finally:
            try:
                await page.close()  # Close page only, keep context alive for session
            except Exception:
                pass
        
        if not manga_title or not image_urls:
            print(f"No images found for {url}")
            return None
        
        return await download_and_merge(
            download_dir=download_dir,
            manga_title=manga_title,
            chapter_num=str(chapter_num),
            image_urls=image_urls,
            referer=referer,
            config=config,
        )


async def process_chapter(url, browser, config, semaphore) -> Optional[str]:
    """
    Process a single chapter: download images and create merged PNG.
    Returns the chapter root path.
    """
    async with semaphore:  # Limit concurrent tabs
        download_dir = config.get("download_dir", "Downloads")

        if is_blocked_toki_url(url):
            print(f"Blocked URL (Newtoki/Manatoki/Booktoki are prohibited): {url}")
            return None
        
        # Determine site type
        site_type = get_site_handler(url)
        if not site_type:
            print(f"Unsupported URL: {url}")
            return None
        
        # Context per chapter for isolation
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            # Extract based on site type
            if site_type == "mangalib":
                manga_title, chapter_num, image_urls = await extract_mangalib_images(page, url)
                referer = "https://mangalib.me/"
            elif site_type == "webtoons":
                manga_title, chapter_num, image_urls = await extract_webtoons_images(page, url)
                referer = "https://www.webtoons.com/"
            elif site_type == "demonic":
                manga_title, chapter_num, image_urls = await extract_demonic_images(page, url)
                referer = url
            else:
                print(f"Unsupported URL: {url}")
                return None
        finally:
            try:
                await context.close()
            except Exception:
                pass  # Context may already be closed
        
        if not manga_title or not image_urls:
            print(f"No images found for {url}")
            return None

        return await download_and_merge(
            download_dir=download_dir,
            manga_title=manga_title,
            chapter_num=str(chapter_num),
            image_urls=image_urls,
            referer=referer,
            config=config,
        )


def archive_title_folders(chapter_paths: List[str]) -> List[str]:
    """Create a .zip for each title folder that has downloaded chapters."""
    title_dirs = sorted({Path(p).parent for p in chapter_paths})
    archives: List[str] = []

    for title_dir in title_dirs:
        if not title_dir.is_dir():
            continue

        try:
            archive_path = shutil.make_archive(
                base_name=str(title_dir),
                format="zip",
                root_dir=str(title_dir.parent),
                base_dir=title_dir.name,
            )
            archives.append(archive_path)
        except Exception as e:
            print(f"Failed to create archive for {title_dir}: {e}")

    return archives


async def main():
    config = load_config()
    urls = config.get("urls", [])
    
    if not urls:
        print("No URLs found in config.yaml. Please add them under the 'urls:' section.")
        return
    
    # Remove duplicates and empty strings
    urls = list(dict.fromkeys([u.strip() for u in urls if u and u.strip()]))
    
    print(f"Found {len(urls)} URLs to process from config.yaml.")
    
    # Categorize URLs by site type
    naver_urls = []
    other_urls = []
    blocked_urls = []
    
    for url in urls:
        if is_blocked_toki_url(url):
            blocked_urls.append(url)
            print(f"Blocked URL (skipping): {url}")
            continue

        site_type = get_site_handler(url)
        if site_type is None:
            print(f"Unsupported URL (skipping): {url}")
        elif site_type == "naver":
            naver_urls.append(url)
        else:
            other_urls.append(url)
    
    if blocked_urls:
        print(f"  - Blocked URLs: {len(blocked_urls)} (Newtoki/Manatoki/Booktoki)")
    print(f"  - Naver URLs: {len(naver_urls)} (with persistent session)")
    print(f"  - Other URLs: {len(other_urls)} (DemonicScans/Mangalib/Webtoons)")
    
    chapter_paths = []
    
    # Semaphore to limit concurrent browser tabs
    max_concurrent = config.get("max_concurrent_chapters", 3)
    print(f"Max concurrent processing: {max_concurrent} chapters")
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # ==================== NAVER PROCESSING (Persistent Context) ====================
    if naver_urls:
        print(f"\n{'='*60}")
        print("Processing Naver URLs (with persistent session)")
        print(f"{'='*60}")
        
        async with async_playwright() as p:
            # Use persistent context for Naver (preserves cookies, localStorage, etc.)
            naver_context = await ensure_naver_login(p, config)
            
            try:
                # Process all Naver chapters using the same authenticated context
                tasks = [process_naver_chapter(url, naver_context, config, semaphore) for url in naver_urls]
                results = await asyncio.gather(*tasks)
                chapter_paths.extend([path for path in results if path is not None])
            finally:
                await naver_context.close()
    
    # ==================== OTHER SITES PROCESSING (Regular Browser) ====================
    if other_urls:
        print(f"\n{'='*60}")
        print("Processing Other URLs (DemonicScans/Mangalib/Webtoons)")
        print(f"{'='*60}")
        
        async with async_playwright() as p:
            # Configure browser launch options with proxy if available
            use_browser_proxy = config.get("use_proxy_for_browser", False)
            proxy_url = config.get("proxy", "") if use_browser_proxy else ""
            headless_mode = config.get("headless", True)
            launch_args = {"headless": headless_mode}
            
            if proxy_url:
                # Parse proxy URL: http://user:pass@host:port
                proxy_match = re.match(r'http://([^:]+):([^@]+)@([^:]+):(\d+)', proxy_url)
                if proxy_match:
                    proxy_user, proxy_pass, proxy_host, proxy_port = proxy_match.groups()
                    print(f"Using proxy for browser: {proxy_host}:{proxy_port}")
                    launch_args["proxy"] = {
                        "server": f"http://{proxy_host}:{proxy_port}",
                        "username": proxy_user,
                        "password": proxy_pass
                    }
                else:
                    print(f"Using proxy for browser: {proxy_url}")
                    launch_args["proxy"] = {"server": proxy_url}
            else:
                print("Running browser without proxy (direct connection)")
            
            browser = await p.chromium.launch(**launch_args)
            
            # Download all chapters and collect paths
            tasks = [process_chapter(url, browser, config, semaphore) for url in other_urls]
            results = await asyncio.gather(*tasks)
            
            await browser.close()
        
        # Collect successful chapter paths
        chapter_paths.extend([path for path in results if path is not None])
    
    print(f"\nAll downloads complete. {len(chapter_paths)} chapters merged.")

    if chapter_paths:
        archives = archive_title_folders(chapter_paths)
        if archives:
            print(f"Archives created: {len(archives)}")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
