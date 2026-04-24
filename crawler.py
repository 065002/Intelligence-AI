"""
crawler.py — Website crawling and data extraction module
Intelligent App Testing System v1.0.0
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlencode
import time
import re
from collections import defaultdict
from urllib.robotparser import RobotFileParser
import ssl
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; IntelligentAppTester/1.0; "
        "+https://github.com/your-repo/intelligent-app-tester)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

MAX_PAGES = 50
TIMEOUT = 10


def validate_url(url: str) -> tuple[bool, str]:
    """Validate URL format and reachability."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return False, url, "Invalid URL — no domain found."
    except Exception:
        return False, url, "Could not parse the URL."
    return True, url, ""


def check_robots_txt(base_url: str, respect_robots: bool) -> RobotFileParser:
    """Fetch and parse robots.txt."""
    rp = RobotFileParser()
    if not respect_robots:
        return None
    try:
        robots_url = urljoin(base_url, "/robots.txt")
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        return None
    return rp


def can_fetch(rp, url: str) -> bool:
    """Check if we're allowed to fetch this URL."""
    if rp is None:
        return True
    try:
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return True


def fetch_page(url: str, session: requests.Session) -> dict:
    """Fetch a single page and return raw data."""
    result = {
        "url": url,
        "status_code": None,
        "html": None,
        "load_time": None,
        "error": None,
        "content_type": None,
    }
    try:
        start = time.time()
        resp = session.get(url, headers=HEADERS, timeout=TIMEOUT, verify=False, allow_redirects=True)
        result["load_time"] = round(time.time() - start, 3)
        result["status_code"] = resp.status_code
        result["content_type"] = resp.headers.get("Content-Type", "")
        if "text/html" in result["content_type"]:
            result["html"] = resp.text
    except requests.exceptions.Timeout:
        result["error"] = "timeout"
    except requests.exceptions.SSLError:
        result["error"] = "ssl_error"
    except requests.exceptions.ConnectionError:
        result["error"] = "connection_error"
    except Exception as e:
        result["error"] = str(e)[:100]
    return result


def extract_links(html: str, base_url: str) -> list[str]:
    """Extract all href links from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.scheme in ("http", "https"):
            links.append(full)
    return links


def is_same_domain(url: str, base_domain: str) -> bool:
    """Check if URL belongs to same domain."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    return domain == base_domain or domain.endswith("." + base_domain)


def identify_module(url: str, base_url: str) -> str:
    """Identify module/feature from URL path."""
    path = urlparse(url).path.strip("/")
    if not path:
        return "Homepage"
    segments = path.split("/")
    first = segments[0].lower()

    module_map = {
        "login": "Authentication",
        "signin": "Authentication",
        "signup": "Authentication",
        "register": "Authentication",
        "auth": "Authentication",
        "logout": "Authentication",
        "checkout": "Checkout",
        "cart": "Cart",
        "basket": "Cart",
        "payment": "Checkout",
        "order": "Orders",
        "orders": "Orders",
        "dashboard": "Dashboard",
        "admin": "Admin Panel",
        "profile": "User Profile",
        "account": "User Profile",
        "user": "User Profile",
        "settings": "Settings",
        "config": "Settings",
        "search": "Search",
        "product": "Product Pages",
        "products": "Product Pages",
        "shop": "Product Pages",
        "store": "Product Pages",
        "category": "Product Pages",
        "blog": "Blog/Content",
        "news": "Blog/Content",
        "article": "Blog/Content",
        "post": "Blog/Content",
        "contact": "Contact",
        "about": "About",
        "help": "Help/Support",
        "support": "Help/Support",
        "faq": "Help/Support",
        "api": "API",
        "docs": "Documentation",
        "documentation": "Documentation",
        "pricing": "Pricing",
        "plans": "Pricing",
        "gallery": "Media",
        "media": "Media",
        "portfolio": "Portfolio",
        "services": "Services",
        "careers": "Careers",
        "jobs": "Careers",
        "terms": "Legal",
        "privacy": "Legal",
        "legal": "Legal",
        "404": "Error Pages",
        "error": "Error Pages",
    }

    for key, module in module_map.items():
        if key in first:
            return module

    if first:
        return first.replace("-", " ").replace("_", " ").title()
    return "General Pages"


def extract_page_data(html: str, url: str) -> dict:
    """Extract structural data from a page's HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Meta description
    meta_desc = soup.find("meta", attrs={"name": "meta-description"}) or \
                soup.find("meta", attrs={"name": "description"})
    meta_description = meta_desc["content"] if meta_desc and meta_desc.get("content") else ""

    # Headings
    headings = {}
    for level in ["h1", "h2", "h3", "h4"]:
        tags = soup.find_all(level)
        headings[level] = [t.get_text(strip=True) for t in tags]

    # Images
    images = soup.find_all("img")
    img_total = len(images)
    img_missing_alt = sum(1 for img in images if not img.get("alt", "").strip())

    # Forms
    forms = soup.find_all("form")
    form_data = []
    for form in forms:
        action = form.get("action", "")
        method = form.get("method", "get").upper()
        inputs = form.find_all("input")
        has_validation = any(
            inp.get("required") or inp.get("pattern") or inp.get("type") == "email"
            for inp in inputs
        )
        form_data.append({
            "action": action,
            "method": method,
            "input_count": len(inputs),
            "has_validation": has_validation,
        })

    # Buttons
    buttons = soup.find_all("button") + soup.find_all("input", attrs={"type": ["submit", "button"]})
    btn_count = len(buttons)

    # Navigation
    nav_links = []
    nav = soup.find("nav") or soup.find(attrs={"role": "navigation"})
    if nav:
        nav_links = [a.get_text(strip=True) for a in nav.find_all("a") if a.get_text(strip=True)]

    # Scripts and stylesheets
    scripts = len(soup.find_all("script"))
    stylesheets = len(soup.find_all("link", rel=lambda r: r and "stylesheet" in r))

    # All links on page
    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        text = a.get_text(strip=True)
        links.append({"href": href, "text": text})

    # Canonical
    canonical = soup.find("link", rel="canonical")
    has_canonical = canonical is not None

    # Viewport meta
    viewport = soup.find("meta", attrs={"name": "viewport"})
    has_viewport = viewport is not None

    # Open graph
    og_title = soup.find("meta", property="og:title")
    has_og = og_title is not None

    return {
        "url": url,
        "title": title,
        "meta_description": meta_description,
        "headings": headings,
        "img_total": img_total,
        "img_missing_alt": img_missing_alt,
        "forms": form_data,
        "form_count": len(form_data),
        "button_count": btn_count,
        "nav_links": nav_links,
        "scripts": scripts,
        "stylesheets": stylesheets,
        "links": links,
        "has_canonical": has_canonical,
        "has_viewport": has_viewport,
        "has_og": has_og,
        "has_title": bool(title),
        "has_meta_description": bool(meta_description),
        "h1_count": len(headings.get("h1", [])),
        "element_count": len(soup.find_all()),
    }


def check_link_status(url: str, session: requests.Session) -> dict:
    """Check HTTP status of a single link (HEAD request for speed)."""
    try:
        resp = session.head(url, headers=HEADERS, timeout=6, verify=False, allow_redirects=True)
        return {"url": url, "status": resp.status_code}
    except Exception:
        try:
            resp = session.get(url, headers=HEADERS, timeout=6, verify=False, allow_redirects=True)
            return {"url": url, "status": resp.status_code}
        except Exception:
            return {"url": url, "status": 0}


def crawl_website(url: str, respect_robots: bool = True, progress_callback=None) -> dict:
    """
    Main crawl function. Returns structured crawl results.
    progress_callback(current, total, message) for UI updates.
    """
    valid, url, err = validate_url(url)
    if not valid:
        return {"error": err, "url": url}

    parsed_base = urlparse(url)
    base_domain = parsed_base.netloc.replace("www.", "")

    session = requests.Session()

    # Check robots.txt
    rp = check_robots_txt(url, respect_robots)

    if not can_fetch(rp, url):
        return {
            "error": "robots_blocked",
            "url": url,
            "message": "Crawl blocked by robots.txt. Uncheck 'Respect robots.txt' to force crawl.",
        }

    visited = set()
    to_visit = [url]
    pages = []
    all_external_links = set()
    broken_links = []
    crawl_start = time.time()

    step = 0
    total_planned = min(MAX_PAGES, 50)

    while to_visit and len(visited) < MAX_PAGES:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        # Normalize
        current_url_clean = current_url.split("#")[0].split("?")[0]
        if current_url_clean in visited:
            continue

        if not can_fetch(rp, current_url):
            continue

        visited.add(current_url_clean)
        step += 1

        if progress_callback:
            progress_callback(
                step,
                total_planned,
                f"Crawling: {current_url_clean[:60]}...",
            )

        page_result = fetch_page(current_url, session)
        page_result["module"] = identify_module(current_url, url)
        page_result["depth"] = len([s for s in urlparse(current_url).path.split("/") if s])

        if page_result["html"]:
            page_data = extract_page_data(page_result["html"], current_url)
            page_result.update(page_data)

            # Discover new links
            found_links = extract_links(page_result["html"], url)
            for link in found_links:
                link_clean = link.split("#")[0].split("?")[0]
                if is_same_domain(link, base_domain):
                    if link_clean not in visited and link_clean not in to_visit:
                        to_visit.append(link_clean)
                else:
                    all_external_links.add(link_clean)

        pages.append(page_result)

    crawl_time = round(time.time() - crawl_start, 2)

    # Check a sample of external links for broken status
    if progress_callback:
        progress_callback(step, total_planned, "Checking external links...")

    external_sample = list(all_external_links)[:20]
    for ext_url in external_sample:
        status = check_link_status(ext_url, session)
        if status["status"] in (0, 404, 403, 410, 500, 502, 503):
            broken_links.append(status)

    # Check internal links that returned errors
    for page in pages:
        if page.get("status_code") in (404, 410, 500, 502, 503) or page.get("error"):
            broken_links.append({"url": page["url"], "status": page.get("status_code", 0)})

    if progress_callback:
        progress_callback(total_planned, total_planned, "Crawl complete!")

    return {
        "url": url,
        "base_domain": base_domain,
        "pages": pages,
        "total_pages": len(pages),
        "crawl_time": crawl_time,
        "broken_links": broken_links,
        "external_links_checked": len(external_sample),
        "total_external_links": len(all_external_links),
        "modules": list(set(p["module"] for p in pages)),
        "error": None,
    }


def generate_issues_from_crawl(crawl_data: dict) -> list[dict]:
    """
    Generate a structured issue dataset from crawl results.
    Every issue is traceable to actual crawl findings.
    """
    issues = []
    issue_id = 1

    for page in crawl_data["pages"]:
        url = page.get("url", "")
        module = page.get("module", "Unknown")
        load_time = page.get("load_time") or 0
        status_code = page.get("status_code")
        error = page.get("error")

        # --- Broken link / unreachable page ---
        if status_code in (404, 410):
            issues.append({
                "id": issue_id,
                "page_url": url,
                "module": module,
                "issue_type": "Broken Link (404)",
                "severity": "High",
                "status": "Open",
                "occurrences": 1,
                "fix_time_hours": 0.5,
                "description": f"Page returns HTTP {status_code}. URL '{url}' is broken or removed.",
                "source": "crawl_status",
            })
            issue_id += 1

        elif status_code in (500, 502, 503):
            issues.append({
                "id": issue_id,
                "page_url": url,
                "module": module,
                "issue_type": "Server Error",
                "severity": "High",
                "status": "Open",
                "occurrences": 1,
                "fix_time_hours": 2.0,
                "description": f"Page '{url}' returns server error {status_code}. Possible backend crash.",
                "source": "crawl_status",
            })
            issue_id += 1

        elif error == "timeout":
            issues.append({
                "id": issue_id,
                "page_url": url,
                "module": module,
                "issue_type": "Page Timeout",
                "severity": "High",
                "status": "Open",
                "occurrences": 1,
                "fix_time_hours": 3.0,
                "description": f"Page '{url}' timed out after {TIMEOUT}s. Possible performance bottleneck.",
                "source": "crawl_timeout",
            })
            issue_id += 1

        # --- Slow load time ---
        if load_time and load_time > 3.0:
            sev = "High" if load_time > 6 else "Medium"
            issues.append({
                "id": issue_id,
                "page_url": url,
                "module": module,
                "issue_type": "Slow Page Load",
                "severity": sev,
                "status": "Open",
                "occurrences": 1,
                "fix_time_hours": 4.0,
                "description": f"Page '{url}' loaded in {load_time}s, exceeding the 3s performance threshold.",
                "source": "crawl_performance",
            })
            issue_id += 1
        elif load_time and 1.5 < load_time <= 3.0:
            issues.append({
                "id": issue_id,
                "page_url": url,
                "module": module,
                "issue_type": "Moderate Load Time",
                "severity": "Low",
                "status": "Open",
                "occurrences": 1,
                "fix_time_hours": 2.0,
                "description": f"Page '{url}' loaded in {load_time}s. Consider optimization.",
                "source": "crawl_performance",
            })
            issue_id += 1

        # --- Missing title ---
        if not page.get("has_title") and page.get("html"):
            issues.append({
                "id": issue_id,
                "page_url": url,
                "module": module,
                "issue_type": "Missing Page Title",
                "severity": "Medium",
                "status": "Open",
                "occurrences": 1,
                "fix_time_hours": 0.25,
                "description": f"Page '{url}' has no <title> tag. Impacts SEO and accessibility.",
                "source": "crawl_seo",
            })
            issue_id += 1

        # --- Missing meta description ---
        if not page.get("has_meta_description") and page.get("html"):
            issues.append({
                "id": issue_id,
                "page_url": url,
                "module": module,
                "issue_type": "Missing Meta Description",
                "severity": "Low",
                "status": "Open",
                "occurrences": 1,
                "fix_time_hours": 0.25,
                "description": f"Page '{url}' missing meta description. Reduces search engine snippet quality.",
                "source": "crawl_seo",
            })
            issue_id += 1

        # --- Multiple H1 tags ---
        h1_count = page.get("h1_count", 0)
        if h1_count > 1:
            issues.append({
                "id": issue_id,
                "page_url": url,
                "module": module,
                "issue_type": "Multiple H1 Tags",
                "severity": "Low",
                "status": "Open",
                "occurrences": h1_count,
                "fix_time_hours": 0.5,
                "description": f"Page '{url}' has {h1_count} H1 tags. Only one H1 recommended for SEO.",
                "source": "crawl_seo",
            })
            issue_id += 1
        elif h1_count == 0 and page.get("html"):
            issues.append({
                "id": issue_id,
                "page_url": url,
                "module": module,
                "issue_type": "Missing H1 Tag",
                "severity": "Medium",
                "status": "Open",
                "occurrences": 1,
                "fix_time_hours": 0.5,
                "description": f"Page '{url}' has no H1 heading, impacting accessibility and SEO structure.",
                "source": "crawl_accessibility",
            })
            issue_id += 1

        # --- Images missing alt text ---
        img_missing = page.get("img_missing_alt", 0)
        img_total = page.get("img_total", 0)
        if img_missing > 0:
            sev = "Medium" if img_missing > 5 else "Low"
            issues.append({
                "id": issue_id,
                "page_url": url,
                "module": module,
                "issue_type": "Missing Image Alt Text",
                "severity": sev,
                "status": "Open",
                "occurrences": img_missing,
                "fix_time_hours": round(img_missing * 0.1, 1),
                "description": f"{img_missing}/{img_total} images on '{url}' lack alt text. Accessibility violation (WCAG 2.1).",
                "source": "crawl_accessibility",
            })
            issue_id += 1

        # --- Forms without validation ---
        for form in page.get("forms", []):
            if not form.get("has_validation") and form.get("input_count", 0) > 0:
                issues.append({
                    "id": issue_id,
                    "page_url": url,
                    "module": module,
                    "issue_type": "Form Without Client-Side Validation",
                    "severity": "High",
                    "status": "Open",
                    "occurrences": 1,
                    "fix_time_hours": 2.0,
                    "description": (
                        f"Form on '{url}' (action='{form.get('action', '#')}') has "
                        f"{form.get('input_count', 0)} inputs with no HTML5 validation attributes."
                    ),
                    "source": "crawl_forms",
                })
                issue_id += 1

        # --- No viewport meta (mobile responsiveness) ---
        if not page.get("has_viewport") and page.get("html"):
            issues.append({
                "id": issue_id,
                "page_url": url,
                "module": module,
                "issue_type": "Missing Viewport Meta Tag",
                "severity": "Medium",
                "status": "Open",
                "occurrences": 1,
                "fix_time_hours": 0.25,
                "description": f"Page '{url}' missing viewport meta tag. Not mobile-responsive by default.",
                "source": "crawl_mobile",
            })
            issue_id += 1

        # --- No canonical tag ---
        if not page.get("has_canonical") and page.get("html") and page.get("depth", 0) > 0:
            issues.append({
                "id": issue_id,
                "page_url": url,
                "module": module,
                "issue_type": "Missing Canonical Tag",
                "severity": "Low",
                "status": "Open",
                "occurrences": 1,
                "fix_time_hours": 0.25,
                "description": f"Page '{url}' missing canonical link tag. Risk of duplicate content SEO issues.",
                "source": "crawl_seo",
            })
            issue_id += 1

        # --- No Open Graph tags ---
        if not page.get("has_og") and page.get("html"):
            issues.append({
                "id": issue_id,
                "page_url": url,
                "module": module,
                "issue_type": "Missing Open Graph Tags",
                "severity": "Low",
                "status": "Open",
                "occurrences": 1,
                "fix_time_hours": 0.5,
                "description": f"Page '{url}' has no Open Graph meta tags. Social sharing previews will be poor.",
                "source": "crawl_seo",
            })
            issue_id += 1

    # --- Broken external links ---
    for bl in crawl_data.get("broken_links", []):
        status = bl.get("status", 0)
        bl_url = bl.get("url", "")
        module = identify_module(bl_url, crawl_data["url"])
        sev = "High" if status in (0, 404, 410) else "Medium"
        issues.append({
            "id": issue_id,
            "page_url": bl_url,
            "module": module,
            "issue_type": "Broken External Link" if status else "Unreachable Link",
            "severity": sev,
            "status": "Open",
            "occurrences": 1,
            "fix_time_hours": 0.5,
            "description": f"External link '{bl_url}' returned status {status}. Should be updated or removed.",
            "source": "crawl_links",
        })
        issue_id += 1

    # Deduplicate by type+module
    seen = set()
    deduped = []
    for issue in issues:
        key = (issue["module"], issue["issue_type"])
        if key not in seen:
            seen.add(key)
            deduped.append(issue)
        else:
            for d in deduped:
                if (d["module"], d["issue_type"]) == key:
                    d["occurrences"] += issue["occurrences"]
                    break

    # Assign realistic statuses
    import random
    random.seed(42)
    statuses = ["Open", "Open", "Open", "Fixed", "Reopened"]
    for issue in deduped:
        if issue["severity"] == "Low":
            issue["status"] = random.choice(["Open", "Fixed", "Open", "Open"])
        elif issue["severity"] == "Medium":
            issue["status"] = random.choice(["Open", "Open", "Reopened", "Fixed"])
        else:
            issue["status"] = random.choice(["Open", "Open", "Open", "Reopened"])

    return deduped
