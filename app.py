from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
import pandas as pd
import praw
import re
from collections import Counter

app = Flask(__name__)
CORS(app)

REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', 'PH99oWZjM43GimMtYigFvA')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '3tJsXQKEtFFYInxzLEDqRZ0s_w5z0g')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'ow_part1_box1_script')

nlp = None

def get_nlp():
    global nlp
    if nlp is None:
        import spacy
        print("Loading spaCy model...")
        nlp = spacy.load("en_core_web_sm")
        print("spaCy model loaded")
    return nlp

print("Connecting to Reddit...")
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    check_for_async=False
)
print("Reddit client ready")

MKT_SEO_SL_SUBS: List[str] = [
    "marketing",
    "SEO",
    "digital_marketing",
    "socialmedia",
    "PPC",
]

bad_org = {
    "Reddit", "YouTube", "Instagram", "TikTok",
    "GOP", "Democrats", "Republicans",
}


def fetch_hot_posts(
    subreddits: List[str],
    days: int = 7,
    per_sub_limit: int = 200,
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows: List[Dict[str, Any]] = []

    for sub in subreddits:
        try:
            subreddit = reddit.subreddit(sub)

            for post in subreddit.hot(limit=per_sub_limit):
                created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)

                if created < cutoff:
                    break

                rows.append({
                    "subreddit": sub,
                    "post_id": post.id,
                    "created_utc": post.created_utc,
                    "created_dt": created.isoformat(),
                    "title": (post.title or "").strip(),
                    "selftext": (post.selftext or "").strip(),
                    "url": getattr(post, "url", ""),
                    "permalink": f"https://www.reddit.com{getattr(post, 'permalink', '')}",
                    "score": getattr(post, "score", None),
                    "num_comments": getattr(post, "num_comments", None),
                    "author": str(getattr(post, "author", "")) if getattr(post, "author", None) else None,
                    "is_self": getattr(post, "is_self", None),
                    "over_18": getattr(post, "over_18", None),
                })

                if sleep_seconds:
                    time.sleep(sleep_seconds)

        except Exception as e:
            print(f"[WARN] Subreddit '{sub}' failed: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["full_text"] = (
        df["title"].astype(str).str.strip()
        + "\n"
        + df["selftext"].astype(str).str.strip()
    ).str.strip()

    df = df.drop_duplicates(subset=["post_id"]).reset_index(drop=True)
    df = df.sort_values("created_utc", ascending=False).reset_index(drop=True)

    return df


def extract_orgs(text: str) -> List[str]:
    
    if not isinstance(text, str) or not text.strip():
        return []

    nlp = get_nlp()
    doc = nlp(text[:5000])
    orgs: List[str] = []

    for ent in doc.ents:
        if ent.label_ == "ORG":
            name = ent.text.strip()
            name = re.sub(r"\s+", " ", name)
            name = name.strip(".,:;()[]{}\"'")

            if len(name) < 2:
                continue
            if name in bad_org:
                continue
            if name.isupper() and len(name) <= 3:
                continue

            orgs.append(name)
    
    return orgs


def collapse_to_parent(org: str, org_set: set) -> str:
    
    s = re.sub(r"\s+", " ", org).strip().strip(".,:;()[]{}\"'")
    s_cf = s.casefold()
    parts = s.split()

    if len(parts) == 1:
        return s_cf

    for k in range(len(parts), 0, -1):
        prefix = " ".join(parts[:k]).casefold()
        if prefix in org_set:
            return prefix

    return parts[0].casefold()


def analyze_companies(days: int = 7, per_sub_limit: int = 200):
    
    df_mkt = fetch_hot_posts(
        MKT_SEO_SL_SUBS, 
        days=days, 
        per_sub_limit=per_sub_limit, 
        sleep_seconds=0.0
    )
    df_mkt["bucket"] = "marketing"

    if df_mkt.empty:
        return {
            "error": "No posts collected",
            "message": "Try increasing per_sub_limit or add more subreddits"
        }

    df_mkt["orgs"] = df_mkt["full_text"].apply(extract_orgs)

    exploded = df_mkt.explode("orgs").dropna(subset=["orgs"]).copy()

    if exploded.empty:
        return {
            "error": "No organizations found",
            "message": "No ORG entities found. Try increasing PER_SUB_LIMIT."
        }

    exploded["org_clean"] = (
        exploded["orgs"].astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.strip(".,:;()[]{}\"'")
    )

    org_set = set(exploded["org_clean"].str.casefold().tolist())

    exploded["company_key"] = exploded["org_clean"].apply(
        lambda org: collapse_to_parent(org, org_set)
    )

    display_map = (
        exploded.assign(company_key=exploded["company_key"])
        .groupby(["company_key", "org_clean"])
        .size()
        .reset_index(name="n")
        .sort_values(["company_key", "n"], ascending=[True, False])
        .drop_duplicates("company_key")
        .set_index("company_key")["org_clean"]
        .to_dict()
    )

    counts = exploded.groupby("company_key").size().sort_values(ascending=False)

    top10 = counts.head(10).reset_index()
    top10.columns = ["company_key", "mentions"]
    top10["company"] = top10["company_key"].map(display_map).fillna(top10["company_key"])
    top10 = top10[["company", "mentions"]]

    return top10


@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "Feature 1 - Companies Needing Social Listening",
        "endpoints": {
            "/": "Health check",
            "/analyze": "Run analysis (GET)",
            "/health": "Service health"
        }
    })


@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()})


@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "status": "working",
        "message": "API is functioning correctly",
        "endpoints": {
            "/": "API info",
            "/health": "Health check", 
            "/test": "Quick test (this endpoint)",
            "/analyze": "Run analysis (takes 2-5 minutes)",
            "/analyze?limit=20": "Fast analysis with fewer posts"
        },
        "tip": "Use /analyze?limit=20 for faster results on free tier"
    })


@app.route('/analyze', methods=['GET'])
def analyze():
    try:
        days = int(request.args.get('days', 7))
        limit = int(request.args.get('limit', 50))
        
        print(f"Starting analysis (days={days}, limit={limit})")
        
        result = analyze_companies(days=days, per_sub_limit=limit)
        
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 400
        
        ranked_companies = []
        for idx, row in result.iterrows():
            ranked_companies.append({
                "rank": int(idx + 1),
                "company": {
                    "id": f"cmp_{row['company'].lower().replace(' ', '_')}",
                    "name": row['company']
                },
                "volume": {
                    "mentions": int(row['mentions'])
                }
            })
        
        response = {
            "data": {
                "ranked_companies": ranked_companies
            },
            "meta": {
                "total": len(ranked_companies),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "time_range_days": days,
                "posts_per_subreddit": limit
            }
        }
        
        print(f"Analysis complete. Found {len(ranked_companies)} companies")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Analysis failed",
            "message": str(e),
            "tip": "Try reducing the 'limit' parameter (e.g., ?limit=20)"
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
