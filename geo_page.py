import json
from typing import Optional, Dict, List

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI

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
def normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    scheme, rest = url.split("://", 1)
    if rest.startswith("www."):
        rest = rest[4:]
    return f"{scheme}://{rest}"

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
# 3) Requests-only fetcher
# ─────────────────────────────────────────────────────────────
def fetch_webpage_rich_content(url: str, debug: bool) -> Optional[Dict[str, object]]:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        if resp.status_code >= 400:
            st.error(f"❌ requests status {resp.status_code} for {url}")
            return None
        html = resp.text
        if debug:
            st.info(f"✅ requests HTML length: {len(html)}")
        data = parse_html_to_dict(html)
        if len(data["text"].strip()) < 100:
            st.warning("⚠ Page fetched but contains very little text.")
        return data
    except Exception as e:
        st.error(f"❌ requests error: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# 4) Semantic continuum
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
    )
    continuum = resp.choices[0].message.content.strip()
    if debug:
        st.write(f"(hidden continuum) {continuum}")
    return continuum

# ─────────────────────────────────────────────────────────────
# 5) Query fan-out
# ─────────────────────────────────────────────────────────────
def generate_unbranded_queries(prompt: str, debug: bool) -> List[str]:
    sys = "Generate 10 unbranded three-word queries. One per line. No bullets."
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": f"Original prompt: {prompt}"}],
    )
    lines = [l.strip(" .") for l in resp.choices[0].message.content.split("\n") if l.strip()]
    lines = lines[:10]
    if debug:
        st.write(f"Fan-out ({len(lines)}): {lines}")
    return lines

# ─────────────────────────────────────────────────────────────
# 6) Build audit context
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
        "Score the content against each of the criteria areas out of 10, give a one-liner justification, and a one-liner action-to-take.\n"
        "Be professional and objective.\n"
        "</task> \n\n"
        "<scoring calibration> \n"
        f"{theme_calibration}"
        "</scoring calibration> \n\n"
        "<extra input> \n"
        f"Guidance: {audit_scoring_context_json}\n"
        "</extra input> \n\n"
    )

# ─────────────────────────────────────────────────────────────
# 7) Audit run
# ─────────────────────────────────────────────────────────────
def audit_page(url: str, user_query: str, audit_context: str, debug: bool) -> Optional[pd.DataFrame]:
    content = fetch_webpage_rich_content(url, debug)
    if content is None:
        return None
    fan_out = generate_unbranded_queries(user_query, debug)
    text_preview = content["text"][:4000]

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
        )
        payload = json.loads(resp.choices[0].message.content)
        df = pd.DataFrame(payload["criteria"])
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0).clip(0, 10).astype(int)
        return df
    except Exception as e:
        st.error(f"❌ LLM JSON parse error: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# 8) Multi-run + synthesis
# ─────────────────────────────────────────────────────────────
def run_multi(url: str, prompt: str, n: int, audit_context: str, debug: bool) -> pd.DataFrame:
    frames = []
    progress = st.progress(0.0, text="Auditing…")
    for i in range(n):
        df = audit_page(url, prompt, audit_context, debug)
        if df is not None and not df.empty:
            df["Run"] = i + 1
            frames.append(df)
        progress.progress((i + 1) / n, text=f"Auditing… ({i+1}/{n})")
    progress.empty()
    if not frames:
        return pd.DataFrame(columns=["Area", "Score", "Justification", "Action-to-take", "Run"])
    return pd.concat(frames, ignore_index=True)

def synthesize_justifications_and_recommendations(df: pd.DataFrame, debug: bool) -> pd.DataFrame:
    by_area = df.groupby("Area").agg(
        Avg_Score=("Score", "mean"),
        Justifications=("Justification", list),
        Actions=("Action-to-take", list),
    ).reset_index()

    rows = []
    for _, r in by_area.iterrows():
        area = r["Area"]
        avg_score = round(float(r["Avg_Score"]), 2)
        sys = "You are an expert auditor. Align justifications with actionable recommendations."
        usr = f"""
Audit Area: {area}
Justifications: {json.dumps(r['Justifications'], ensure_ascii=False, indent=2)}
Actions: {json.dumps(r['Actions'], ensure_ascii=False, indent=2)}
Return JSON with "justification" and "recommendations" (array of 3-6).
"""
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                response_format={"type": "json_object"},
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
            rows.append({
                "Area": area,
                "Avg Score": avg_score,
                "Synthesized Justification": "Synthesis failed.",
                "Synthesized Recommendation": "",
            })

    return pd.DataFrame(rows).sort_values("Area").reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# 9) Streamlit UI
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
        continuum = generate_theme_calibration(topic, prompt, debug)
        audit_context = build_audit_context(continuum)

    with st.spinner("Running audits…"):
        results_df = run_multi(url, prompt, n_runs, audit_context, debug)

    if results_df.empty:
        st.error("No results returned. Check network or LLM output.")
    else:
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
