import asyncio
import json
from typing import Optional, Dict, List

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# ─────────────────────────────────────────────────────────────
# 0) Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Single Page GEO Audit", page_icon="🔎", layout="centered")

MODEL_NAME = "gpt-5-mini"
AUDIT_AREA_LIST = [
    "Prompt Relevance",
    "Content Depth",
    "Conversational Tone",
    "Internal Linking",
    "EEAT Signals",
    "Personalisation Resilience",
    "Crawlability & Indexability",
]

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ─────────────────────────────────────────────────────────────
# 1) Scoring guidance JSON (kept, verbatim)
# ─────────────────────────────────────────────────────────────
audit_scoring_context_json = {
  "Prompt Relevance": {
    "guidance": "Content aligns with AI search queries: long, conversational, multi-turn, task-oriented.",
    "Good": "Covers branded and unbranded prompts, anticipates varied intents.",
    "Bad": "Focuses only on short keywords, branded queries, ignores AI search patterns."
  },
  "Content Depth": {
    "guidance": "Topical breadth and depth with hub/cluster structure.",
    "Good": "Pillar + cluster pages, cross-linked, each subtopic covered thoroughly.",
    "Bad": "Shallow single-page coverage, no dedicated URLs for subtopics, weak semantic connections."
  },
  "Conversational Tone": {
    "guidance": "Clear, concise, factual, Q&A-friendly, extractable by AI.",
    "Good": "Direct summaries, structured chunks, plain language, structured data.",
    "Bad": "Promotional, vague, multi-concept paragraphs, unclear headings."
  },
  "Internal Linking": {
    "guidance": "Supports crawlability, semantic relationships, hub-and-spoke navigation.",
    "Good": "Descriptive anchor text, crawlable HTML links, meaningful internal connections.",
    "Bad": "Vague anchors ('click here'), JS-only links, weak inter-page links."
  },
  "EEAT Signals": {
    "guidance": "Content is authoritative, accurate, verifiable, up-to-date.",
    "Good": "Named experts, citations, timestamps, original research, structured metadata.",
    "Bad": "Generic byline, no sources, outdated, inaccurate, affiliate-style content."
  },
  "Personalisation Resilience": {
    "guidance": "Remains useful across multiple intents, personas, and localizations.",
    "Good": "Multi-intent coverage, localized content, persona-specific sections, strong engagement, high entity recognition.",
    "Bad": "Single intent, no localization, ignores personas, weak engagement or authority."
  },
  "Crawlability & Indexability": {
    "guidance": "AI and search bots can access, parse, and reuse content.",
    "Good": "AI bots allowed (GPTBot, ClaudeBot, PerplexityBot), semantic HTML, schema, alt text, table markup, canonicalization, metadata-friendly.",
    "Bad": "Blocks AI bots, JS-only content, noindex/nosnippet, duplicate canonicals, unstructured images/tables."
  }
}

# ─────────────────────────────────────────────────────────────
# 2) Utilities
# ─────────────────────────────────────────────────────────────
def dbg(enabled: bool, *args):
    if enabled:
        st.write(*args)

def normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    # Prefer non-www to reduce bot-protection edge cases
    scheme, rest = url.split("://", 1)
    if rest.startswith("www."):
        rest = rest[4:]
    return f"{scheme}://{rest}"

def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

def parse_html_to_dict(html: str) -> Dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title else ""
    meta = soup.find("meta", attrs={"name": "description"})
    meta_desc = meta.get("content", "").strip() if meta else ""
    headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
    links = [(a.get_text(strip=True), a.get("href")) for a in soup.find_all("a", href=True)]
    images = [(img.get("src"), img.get("alt", "")) for img in soup.find_all("img")]
    text = soup.get_text(separator="\n")
    return {
        "title": title,
        "meta_description": meta_desc,
        "headings": headings,
        "links": links,
        "images": images,
        "text": text,
    }

# ─────────────────────────────────────────────────────────────
# 3) Fetchers (Playwright + requests fallback) with visible errors
# ─────────────────────────────────────────────────────────────
async def fetch_with_playwright(url: str, debug: bool) -> Optional[str]:
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.set_extra_http_headers({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            })
            try:
                await page.goto(url, timeout=45000)
                await page.wait_for_selector("body", timeout=15000)
                html = await page.content()
                await browser.close()
                dbg(debug, f"✅ Playwright HTML length: {len(html)}")
                return html
            except PlaywrightTimeoutError:
                st.error(f"⏱ Timeout loading {url}")
                await browser.close()
                return None
            except Exception as e:
                st.error(f"❌ Playwright navigation error: {e}")
                await browser.close()
                return None
    except Exception as e:
        st.error(f"❌ Playwright launch error: {e}")
        return None

def fetch_with_requests(url: str, debug: bool) -> Optional[str]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"},
            timeout=25,
            allow_redirects=True,
        )
        if resp.status_code >= 400:
            st.error(f"❌ requests status {resp.status_code} for {url}")
            return None
        dbg(debug, f"✅ requests HTML length: {len(resp.text)}")
        return resp.text
    except Exception as e:
        st.error(f"❌ requests error: {e}")
        return None

async def fetch_webpage_rich_content(url: str, debug: bool) -> Optional[Dict[str, object]]:
    st.info(f"🌐 Fetching: {url}")
    html = await fetch_with_playwright(url, debug)
    if html is None:
        st.warning("Playwright failed; trying requests fallback…")
        html = fetch_with_requests(url, debug)
        if html is None:
            return None
    data = parse_html_to_dict(html)
    if len(data["text"].strip()) < 100:
        st.warning("⚠ Page fetched but contains very little text (JS-heavy or thin content).")
    dbg(debug, f"Title: {data['title']}")
    dbg(debug, f"Meta: {data['meta_description'][:160]}…")
    dbg(debug, f"Text chars: {len(data['text'])}")
    return data

# ─────────────────────────────────────────────────────────────
# 4) Hidden: semantic continuum (theme calibration)
# ─────────────────────────────────────────────────────────────
def generate_theme_calibration(topic: str, prompt: str, debug: bool) -> str:
    sys = "You are an expert in semantic similarity and thematic relevance."
    usr = f"""
Given the topic: "{topic}"
and the user search prompt: "{prompt}"
Generate a concise continuum of example domains from 0.0 (irrelevant) to 1.0 (perfect match).
Return 1–3 lines only.
"""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        temperature=0.7,
    )
    continuum = resp.choices[0].message.content.strip()
    dbg(debug, f"(hidden continuum) {continuum}")
    return continuum

# ─────────────────────────────────────────────────────────────
# 5) Query fan-out (unbranded variants)
# ─────────────────────────────────────────────────────────────
def generate_unbranded_queries(prompt: str, debug: bool) -> List[str]:
    sys = "Generate 10 unbranded three-word queries. One per line. No bullets."
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": f"Original prompt: {prompt}"}],
        temperature=0.2,
    )
    lines = [l.strip(" .") for l in resp.choices[0].message.content.split("\n") if l.strip()]
    lines = lines[:10]
    dbg(debug, f"Fan-out ({len(lines)}): {lines}")
    return lines

# ─────────────────────────────────────────────────────────────
# 6) Build audit context (kept, with continuum applied)
# ─────────────────────────────────────────────────────────────
def build_audit_context(theme_calibration: str) -> str:
    return (
        "<system> \n"
        "You are a site content auditor. Your job is to evaluate whether a given webpage "
        "provides a clear, complete, and useful response to a specific user query, specifically in the context "
        "of Answer Engine Optimization (AEO).\n"
        "</system> \n\n"
        "<objective> \n"
        "Your goal is to audit the webpage content and surface improvement suggestions for quick wins which will improve brand visibility to LLMs.\n"
        "Evaluate the page based on how relevant the content is in regards to the intent of the user query (search prompt), "
        "given specific considerations you need to bear in mind for maximising visibility to LLMs. \n"
        "</objective> \n\n"
        "<task> \n"
        "Your audit should return the following:\n"
        "Score the content against each of the critera areas out of 10 (integer values only) using the provided scoring guidance for each area, "
        "and strictly within the context and perceived intent of the user query. "
        "Generate also a one-liner justification for the score given.\n "
        "And finally a one-liner action-to-take for a quick win improvement to the site content.\n"
        "Be sure to keep a professional tone and be objective. \n"
        "</task> \n\n"
        "<scoring calibration> \n"
        "A score of 0 reflects that none of the good examples are present, and many of the bad examples are present, "
        "or that the content is not relevant in answering the user query. "
        "A score of 10 reflects that most or all of the good examples are present, and none or very few of the bad examples are present.\n"
        "Where the theme or subject of the user query intent and the theme of the content does not align, all area scores shall be "
        "detrimented, even if good examples exist for that area. Do this as follows: \n"
        "To each score, apply a multiplier (between 0 and 1) based specifically on the relevance of the content theme to "
        "the theme of the user search query. Use the following examples of theme alignment to help semantically determine this multiplier: \n"
        f"{theme_calibration}"
        "</scoring calibration> \n\n"
        "<extra input> \n"
        "Here is a json object detailing guidance for each of the areas to critique. "
        f"Guidance: {audit_scoring_context_json}\n\n"
        "To help you determine the intent of the user search query you will also be provided with a query fan-out "
        "comprising of 10 sets of related keywords as part of each request.\n"
        "</extra input> \n\n"
    )

# ─────────────────────────────────────────────────────────────
# 7) One audit run (strict JSON schema, 7 criteria)
# ─────────────────────────────────────────────────────────────
async def audit_page(url: str, user_query: str, audit_context: str, debug: bool) -> Optional[pd.DataFrame]:
    content = await fetch_webpage_rich_content(url, debug)
    if content is None:
        return None

    fan_out = generate_unbranded_queries(user_query, debug)
    text_preview = content["text"][:4000]  # keep prompt size sane

    user_prompt = f"""
Run a page audit for LLM/GEO visibility.

User query: {user_query}

Query fan-out: {fan_out}

Page title: {content['title']}
Meta description: {content['meta_description']}
Headings: {content['headings'][:20]}
First 30 links: {content['links'][:30]}

Page text (truncated):
{text_preview}
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": audit_context},
                      {"role": "user", "content": user_prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "criteria_audit",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "criteria": {
                                "type": "array",
                                "minItems": 7,
                                "maxItems": 7,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "Area": {"type": "string", "enum": AUDIT_AREA_LIST},
                                        "Score": {"type": "integer", "minimum": 0, "maximum": 10},
                                        "Justification": {"type": "string"},
                                        "Action-to-take": {"type": "string"}
                                    },
                                    "required": ["Area", "Score", "Justification", "Action-to-take"]
                                }
                            }
                        },
                        "required": ["criteria"]
                    }
                }
            },
            temperature=0.2,
        )
        payload = json.loads(resp.choices[0].message.content)
        df = pd.DataFrame(payload["criteria"])
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0).clip(0, 10).astype(int)
        return df
    except Exception as e:
        st.error(f"❌ LLM JSON parse error: {e}")
        try:
            st.code(resp.choices[0].message.content, language="json")
        except Exception:
            pass
        return None

# ─────────────────────────────────────────────────────────────
# 8) Multi-run orchestration
# ─────────────────────────────────────────────────────────────
async def run_multi(url: str, prompt: str, n: int, audit_context: str, debug: bool) -> pd.DataFrame:
    frames = []
    progress = st.progress(0.0, text="Auditing…")
    for i in range(n):
        dbg(debug, f"—— Run {i+1}/{n} ——")
        df = await audit_page(url, prompt, audit_context, debug)
        if df is not None and not df.empty:
            df["Run"] = i + 1
            frames.append(df)
        progress.progress((i + 1) / n, text=f"Auditing… ({i+1}/{n})")
    progress.empty()
    if not frames:
        return pd.DataFrame(columns=["Area", "Score", "Justification", "Action-to-take", "Run"])
    return pd.concat(frames, ignore_index=True)

# ─────────────────────────────────────────────────────────────
# 9) Synthesis: align justifications with recommendations
# ─────────────────────────────────────────────────────────────
def synthesize_justifications_and_recommendations(df: pd.DataFrame, debug: bool) -> pd.DataFrame:
    by_area = (
        df.groupby("Area")
        .agg(
            Avg_Score=("Score", "mean"),
            Justifications=("Justification", list),
            Actions=("Action-to-take", list),
        )
        .reset_index()
    )

    rows = []
    for _, r in by_area.iterrows():
        area = r["Area"]
        avg_score = round(float(r["Avg_Score"]), 2)
        sys = "You are an expert auditor. Align justifications with actionable recommendations."
        usr = f"""
Audit Area: {area}

Combine and ALIGN these justifications and actions:
Justifications: {json.dumps(r['Justifications'], ensure_ascii=False, indent=2)}
Actions: {json.dumps(r['Actions'], ensure_ascii=False, indent=2)}

Rules:
- Produce a single concise justification (2–4 sentences) that logically supports the actions.
- Produce 3–6 concrete, non-duplicative, prioritized actions.
- Ensure every action is defensible by the justification.

Return JSON:
{{
  "justification": "string",
  "recommendations": ["bullet 1", "bullet 2", ...]
}}
"""
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            data = json.loads(resp.choices[0].message.content)
            recs = data.get("recommendations", [])
            recs_txt = "\n".join(f"- {x}" for x in recs) if isinstance(recs, list) else str(recs)
            rows.append({
                "Area": area,
                "Avg Score": avg_score,
                "Synthesized Justification": data.get("justification", "").strip(),
                "Synthesized Recommendation": recs_txt.strip(),
            })
        except Exception as e:
            dbg(debug, f"Synthesis error for {area}: {e}")
            rows.append({
                "Area": area,
                "Avg Score": avg_score,
                "Synthesized Justification": "Synthesis failed.",
                "Synthesized Recommendation": "",
            })

    out = pd.DataFrame(rows).sort_values("Area").reset_index(drop=True)
    return out

# ─────────────────────────────────────────────────────────────
# 10) UI
# ─────────────────────────────────────────────────────────────
st.title("🔎 Single Page GEO Audit")
st.caption("Analyze a single page for LLM/GEO readiness.")

colL, colR = st.columns([3, 1])
with colL:
    url_input = st.text_input("Page URL", value="https://www.example.com")
with colR:
    debug = st.toggle("Debug logs", value=False)

prompt = st.text_area("Custom Prompt", value="what is the best digital business transformation company?")
topic = st.text_input("Topic", value="IT consultancy")
n_runs = st.slider("Number of Runs", min_value=1, max_value=10, value=3)

if st.button("Run Analysis", type="primary"):
    url = normalize_url(url_input)
    with st.spinner("Calibrating…"):
        continuum = generate_theme_calibration(topic, prompt, debug)  # hidden
        audit_context = build_audit_context(continuum)

    with st.spinner("Running audits…"):
        results_df = run_async(run_multi(url, prompt, n_runs, audit_context, debug))

    if results_df.empty:
        st.error("No results returned. Check the visible errors above (scrape/JSON). Try a simpler URL like https://example.com.")
    else:
        # Average overall score across all runs/areas
        overall_avg_pct = (results_df["Score"].mean() / 10.0) * 100.0

        st.subheader("Run Summary")
        st.table(pd.DataFrame([{
            "URL": url,
            "Prompt": prompt,
            "Topic": topic,
            "n_runs": n_runs,
            "Average Overall Score": f"{overall_avg_pct:.1f}%"
        }]))

        st.subheader("Consolidated Results (7 areas)")
        consolidated = synthesize_justifications_and_recommendations(results_df, debug)
        st.dataframe(consolidated, use_container_width=True)
