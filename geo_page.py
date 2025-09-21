import streamlit as st
import pandas as pd
import json
import asyncio
from openai import OpenAI
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup

# --- API Key ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Config ---
MODEL_NAME = "gpt-5-mini"
PRODUCT_AREA = "vehicle breakdown cover"

AUDIT_AREA_LIST = [
    "Prompt Relevance",
    "Content Depth",
    "Conversational Tone",
    "Internal Linking",
    "EEAT Signals",
    "Personalisation Resilience",
    "Crawlability & Indexability"
]

# --- Functions from Notebook (trimmed but intact) ---
async def fetch_webpage_rich_content(url: str) -> dict | None:
    try:
        async with async_playwright() as p:
            browser = await (p.chromium if "greenflag" in url else p.firefox).launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=30000)
            await page.wait_for_load_state("networkidle")
            html = await page.content()
            await browser.close()
    except Exception:
        return None

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title else ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    meta_desc = meta_tag["content"].strip() if meta_tag and "content" in meta_tag.attrs else ""
    headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
    links = [(a.get_text(strip=True), a.get("href")) for a in soup.find_all("a", href=True)]
    images = [(img.get("src"), img.get("alt", "")) for img in soup.find_all("img")]
    text = soup.get_text(separator="\n")

    return {"title": title, "meta_description": meta_desc, "headings": headings,
            "links": links, "images": images, "text": text}


def generate_theme_calibration(topic: str, prompt: str) -> str:
    system_msg = "You are an expert in semantic similarity and thematic relevance."
    user_msg = f"""
Given the topic: "{topic}"
and the user search prompt: "{prompt}"

Generate a continuum of 6-8 example domains from 0.0 (irrelevant)
to 1.0 (perfect match). Format like:
"Domain: 0.0, Domain: 0.2, ..., Domain: 1.0"
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_msg}],
        temperature=1
    )
    return response.choices[0].message.content.strip()


def generate_unbranded_queries(prompt: str) -> list[str]:
    system_message = "<system>You are a helpful assistant.</system>"
    user_message = f"Original prompt: {prompt}\nReturn exactly 10 lines of 3-word search queries."
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system_message},
                  {"role": "user", "content": user_message}]
    )
    output_text = response.choices[0].message.content.strip()
    queries = [line.strip(" .") for line in output_text.split("\n") if line.strip()]
    return queries[:10]


async def audit_page(url: str, user_query: str, product: str, audit_context: str):
    content = await fetch_webpage_rich_content(url)
    if content is None:
        return None

    fan_out_result = generate_unbranded_queries(user_query)

    user_prompt = f"""
Run an audit upon the following:

The user query: {user_query}

Query fan-out: {fan_out_result}

Page title: {content['title']}
Meta description: {content['meta_description']}
Headings: {content['headings']}
Links: {content['links']}
Images: {content['images']}

Page text:
{content['text']}
"""
    response = client.chat.completions.create(
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
                            "items": {
                                "type": "object",
                                "properties": {
                                    "Area": {"type": "string", "enum": AUDIT_AREA_LIST},
                                    "Score": {"type": "integer"},
                                    "Justification": {"type": "string"},
                                    "Action-to-take": {"type": "string"}
                                },
                                "required": ["Area", "Score", "Justification", "Action-to-take"]
                            },
                            "minItems": 7,
                            "maxItems": 7
                        }
                    },
                    "required": ["criteria"]
                }
            }
        }
    )
    parsed = json.loads(response.choices[0].message.content)
    return parsed["criteria"]


def to_df(audit_json: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(audit_json)
    df['Score'] = df['Score'].astype(int)
    return df


async def run_multi(url, prompt, n_runs, audit_context):
    frames = []
    for i in range(1, n_runs + 1):
        result = await audit_page(url, prompt, PRODUCT_AREA, audit_context)
        if result is None:
            continue
        df = to_df(result)
        df['Run'] = i
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['Area', 'Score', 'Justification', 'Action-to-take', 'Run'])
    return pd.concat(frames, ignore_index=True)


def synthesize_justifications_and_recommendations(df: pd.DataFrame):
    grouped = df.groupby("Area").agg({
        "Justification": list,
        "Action-to-take": list,
        "Score": "mean"
    }).reset_index()

    results = []
    for _, row in grouped.iterrows():
        area = row["Area"]
        synthesis_prompt = f"""
You are a precise content auditor.

For the audit area: {area}

Combine these justifications into a concise, single high-quality justification:
{json.dumps(row["Justification"], indent=2)}

Combine these action-to-take suggestions into a concise, non-redundant list:
{json.dumps(row["Action-to-take"], indent=2)}

Return a JSON with fields: justification, recommendation (recommendation can be multiple bullets).
"""
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": "You synthesize expert audit results."},
                      {"role": "user", "content": synthesis_prompt}],
            response_format={"type": "json"}
        )
        parsed = json.loads(response.choices[0].message.content)
        results.append({
            "Area": area,
            "Avg Score": round(row["Score"], 2),
            "Synthesized Justification": parsed["justification"],
            "Synthesized Recommendation": "\n".join(parsed["recommendation"]) if isinstance(parsed["recommendation"], list) else parsed["recommendation"]
        })

    return pd.DataFrame(results)


# --- Streamlit UI ---
st.title("🔎 Single Page GEO Audit")
st.markdown("Analyze a single page for LLM/GEO readiness.")

url = st.text_input("Page URL", value="https://www.example.com")
prompt = st.text_area("Custom Prompt", value="Write a meta description for this page")
topic = st.text_input("Topic", value="SEO / GEO Optimization")
n_runs = st.slider("Number of Runs", min_value=1, max_value=10, value=3)

if st.button("Run Analysis"):
    with st.spinner("Running multi-run audit..."):
        THEME_CALIBRATION = generate_theme_calibration(topic, prompt)

        audit_context = f"""
<system>You are a site content auditor.</system>
<objective>Your goal is to evaluate and suggest quick-win improvements for LLM visibility.</objective>
<task>Score each area 0-10, provide justification and actionable improvement.</task>
<scoring calibration>Apply a multiplier based on semantic alignment:
{THEME_CALIBRATION}
</scoring calibration>
"""
        results_df = asyncio.run(run_multi(url, prompt, n_runs, audit_context))

        if results_df.empty:
            st.error("No results returned. Check URL or retry.")
        else:
            avg_overall = (results_df["Score"].sum() / (len(results_df) * 10)) * 100

            st.subheader("Run Summary")
            st.markdown(f"**URL:** {url}  \n**Prompt:** {prompt}  \n**Topic:** {topic}  \n**N Runs:** {n_runs}")
            st.metric("Average Overall Score", f"{avg_overall:.1f}%")

            st.subheader("Consolidated Results")
            synthesized_df = synthesize_justifications_and_recommendations(results_df)
            st.dataframe(synthesized_df, use_container_width=True)
