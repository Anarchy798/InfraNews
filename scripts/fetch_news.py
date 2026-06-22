#!/usr/bin/env python3
"""
Fetch European infrastructure news from RSS feeds,
analyse each article with Claude for PE investment relevance,
and write the results to data/news.json.
"""

import json
import hashlib
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import anthropic
from dateutil import parser as dateutil_parser

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CLAUDE_MODEL    = "claude-haiku-4-5-20251001"
MAX_OUTPUT      = 30
MAX_TO_ANALYSE  = 50

RSS_FEEDS = [
    {"url": "https://www.infrastructureinvestor.com/feed/",        "source": "Infrastructure Investor"},
    {"url": "https://gihub.org/feed/",                              "source": "GI Hub"},
    {"url": "https://www.privateequityinternational.com/feed/",     "source": "PEI Media"},
    {"url": "https://privatequitywire.co.uk/feed/",                 "source": "Private Equity Wire"},
    {"url": "https://nordsip.com/feed/",                            "source": "NordSIP"},
    {"url": "https://www.euractiv.com/section/energy/feed/",        "source": "Euractiv Energy"},
    {"url": "https://www.euractiv.com/section/transport/feed/",     "source": "Euractiv Transport"},
    {"url": "https://www.euractiv.com/section/digital/feed/",       "source": "Euractiv Digital"},
    {"url": "https://www.eib.org/en/rss/press.htm",                 "source": "EIB"},
    {"url": "https://www.ipe.com/rss/news",                         "source": "IPE"},
    {"url": "https://feeds.reuters.com/reuters/businessNews",       "source": "Reuters Business"},
    {"url": "https://www.theguardian.com/business/rss",             "source": "Guardian Business"},
    {"url": "https://www.rechargenews.com/rss",                     "source": "Recharge News"},
    {"url": "https://www.windpowermonthly.com/rss/",                "source": "Wind Power Monthly"},
]

INFRA_KEYWORDS = {
    "infrastructure", "airport", "seaport", "port", "highway", "motorway",
    "railway", "rail", "metro", "energy", "power grid", "wind farm", "offshore wind",
    "solar", "renewable", "hydrogen", "data centre", "data center", "fibre",
    "broadband", "telecom", "water utility", "waste", "sewage", "toll road",
    "bridge", "tunnel", "hospital", "social housing", "school", "acquisition",
    "concession", "ppp", "private equity", "pension fund", "infrastructure fund",
    "greenfield", "brownfield", "asset management", "fund", "deal", "invest",
    "billion", "million", "eur", "gbp",
}

EUROPE_KEYWORDS = {
    "europe", "european", "eu", "uk", "britain", "england", "scotland", "wales",
    "france", "germany", "italy", "spain", "portugal", "netherlands", "belgium",
    "switzerland", "austria", "poland", "czech", "slovakia", "hungary", "romania",
    "bulgaria", "greece", "sweden", "norway", "denmark", "finland", "ireland",
    "luxembourg", "slovenia", "croatia", "serbia", "ukraine", "turkey",
    "scandinavia", "nordic", "iberian", "balkan", "baltic", "benelux",
}

def _lower(text):
    return re.sub(r"[^\w\s]", " ", text.lower())

def is_relevant(title, snippet=""):
    text = _lower(title + " " + snippet)
    words = set(text.split())
    has_infra  = bool(words & INFRA_KEYWORDS or any(k in text for k in INFRA_KEYWORDS if " " in k))
    has_europe = bool(words & EUROPE_KEYWORDS)
    return has_infra or has_europe

def article_id(title, url):
    return hashlib.md5((title + url).encode()).hexdigest()[:12]

def parse_date(entry):
    for field in ("published", "updated", "created"):
        raw = getattr(entry, field, None) or (entry.get(field) if isinstance(entry, dict) else None)
        if raw:
            try:
                return dateutil_parser.parse(raw).astimezone(timezone.utc).isoformat()
            except Exception:
                pass
    return datetime.now(timezone.utc).isoformat()

def strip_html(text):
    return re.sub(r"<[^>]+>", "", text or "").strip()

def fetch_all():
    seen, articles = set(), []
    for feed_cfg in RSS_FEEDS:
        url, source = feed_cfg["url"], feed_cfg["source"]
        log.info("Fetching %s ...", source)
        try:
            feed = feedparser.parse(url, agent="InfraNews/1.0")
            for entry in feed.entries[:20]:
                title   = (entry.get("title") or "").strip()
                link    = entry.get("link") or entry.get("url") or ""
                snippet = strip_html(entry.get("summary") or entry.get("description") or "")
                if not title or not link:
                    continue
                aid = article_id(title, link)
                if aid in seen:
                    continue
                seen.add(aid)
                if not is_relevant(title, snippet):
                    continue
                articles.append({"id":aid,"title":title,"url":link,"source":source,"raw_summary":snippet[:800],"published":parse_date(entry)})
        except Exception as exc:
            log.warning("Failed %s: %s", source, exc)
    articles.sort(key=lambda a: a["published"], reverse=True)
    return articles[:MAX_TO_ANALYSE]

PROMPT_TEMPLATE = """You are a senior analyst at a European infrastructure private equity fund.

Article title   : {title}
Source          : {source}
Published       : {published}
Content snippet : {snippet}

Reply with ONLY a JSON object — no markdown fences, no commentary. Fields:
- "summary"  : 2 sentences, written for a PE investor, highlighting deal/market implications (string)
- "pe_score" : integer 1-10, how relevant this is to a European infra PE investor (10 = essential reading)
- "sector"   : exactly one of: energy | transport | digital | utilities | social | finance
- "country"  : primary European country or region involved, or "Europe" (string)
- "pe_angle" : 3-5 word phrase describing the investment theme, e.g. "Offshore wind M&A" (string)"""

def analyse_article(art, client):
    prompt = PROMPT_TEMPLATE.format(title=art["title"],source=art["source"],published=art["published"],snippet=art["raw_summary"] or "(no snippet)")
    try:
        msg = client.messages.create(model=CLAUDE_MODEL, max_tokens=320, messages=[{"role":"user","content":prompt}])
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n",1)[1].rsplit("```",1)[0].strip()
        analysis = json.loads(raw)
        return {"id":art["id"],"title":art["title"],"url":art["url"],"source":art["source"],"published":art["published"],"summary":str(analysis.get("summary",art["raw_summary"][:200])),"pe_score":max(1,min(10,int(analysis.get("pe_score",5)))),"sector":str(analysis.get("sector","finance")),"country":str(analysis.get("country","Europe")),"pe_angle":str(analysis.get("pe_angle","")),"fetched_at":datetime.now(timezone.utc).isoformat()}
    except Exception as exc:
        log.warning("Claude error for '%s': %s", art["title"][:50], exc)
        return {"id":art["id"],"title":art["title"],"url":art["url"],"source":art["source"],"published":art["published"],"summary":art["raw_summary"][:200],"pe_score":5,"sector":"finance","country":"Europe","pe_angle":"","fetched_at":datetime.now(timezone.utc).isoformat()}

def main():
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        log.error("CLAUDE_API_KEY environment variable is not set")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)
    log.info("=== Fetching RSS feeds ===")
    articles = fetch_all()
    log.info("Found %d relevant articles", len(articles))
    if not articles:
        log.warning("No articles found — exiting")
        return
    log.info("=== Analysing with Claude ===")
    results = []
    for i, art in enumerate(articles):
        log.info("[%d/%d] %s", i+1, len(articles), art["title"][:70])
        results.append(analyse_article(art, client))
    results.sort(key=lambda a: a["pe_score"], reverse=True)
    output = results[:MAX_OUTPUT]
    out_path = Path(__file__).resolve().parent.parent / "data" / "news.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %d articles to %s", len(output), out_path)

if __name__ == "__main__":
    main()
