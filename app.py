from flask import Flask, jsonify, request
from flask_cors import CORS
import praw
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
import os
import warnings
warnings.filterwarnings('ignore')

# NLTK for sentiment (lighter than transformers)
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
CORS(app)

REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', 'PH99oWZjM43GimMtYigFvA')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '3tJsXQKEtFFYInxzLEDqRZ0s_w5z0g')

print("Connecting to Reddit...")
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent='Sentivity_Overwatch_Feature4_API',
    check_for_async=False
)
print("Reddit client ready")

sentiment_analyzer = None

def get_sentiment_analyzer():
    global sentiment_analyzer
    if sentiment_analyzer is None:
        print("Loading sentiment analyzer (NLTK VADER)...")
        sentiment_analyzer = SentimentIntensityAnalyzer()
        print("Sentiment analyzer loaded")
    return sentiment_analyzer

SUBREDDITS = ['marketing', 'SEO', 'digital_marketing', 'socialmedia', 'PPC', 'analytics']
SEARCH_TERMS = ['social listening', 'Brandwatch', 'Brand24', 'Hootsuite', 'Sprout Social', 'customer voice', 'social media monitoring']

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def reddit_influence(row):
    """
    Standardized influence algorithm from example script.
    Uses logarithmic scaling to prevent outliers from dominating.
    """
    score = max(row.get('score', 0) or 0, 0)
    comments = max(row.get('num_comments', 0) or 0, 0)
    return np.log1p(score) + 0.5 * np.log1p(comments)

def collect_reddit_posts(subreddits, search_terms, time_filter='week', limit=20):
    """
    Collect Reddit posts - searches r/all first, then specific subreddits.
    Uses 'relevance' sorting for best quality results.
    """
    all_posts = []
    
    # STEP 1: Search r/all first
    for term in search_terms:
        try:
            print(f"  Searching r/all for '{term}'...", end=' ')
            sub = reddit.subreddit('all')
            posts = sub.search(term, sort='relevance', time_filter=time_filter, limit=limit)
            
            count = 0
            for post in posts:
                post.comments.replace_more(limit=0)
                comments = [c.body for c in post.comments.list()[:5]]
                
                all_posts.append({
                    'id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                    'url': post.url,
                    'source': 'r/all',
                    'top_comments': ' '.join(comments)
                })
                count += 1
            print(f'Found {count} posts')
        except Exception as e:
            print(f'Error: {e}')
    
    # STEP 2: Search specific subreddits
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        
        for term in search_terms:
            try:
                print(f"  Searching r/{subreddit_name} for '{term}'...", end=' ')
                posts = subreddit.search(term, sort='relevance', time_filter=time_filter, limit=limit)
                
                count = 0
                for post in posts:
                    post.comments.replace_more(limit=0)
                    comments = [c.body for c in post.comments.list()[:5]]
                    
                    all_posts.append({
                        'id': post.id,
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                        'url': post.url,
                        'source': f'r/{subreddit_name}',
                        'top_comments': ' '.join(comments)
                    })
                    count += 1
                print(f'Found {count} posts')
            except Exception as e:
                print(f'Error: {e}')
                continue
    
    df = pd.DataFrame(all_posts)
    
    # Deduplicate by post ID
    if len(df) > 0:
        initial_count = len(df)
        df = df.drop_duplicates(subset=['id'])
        final_count = len(df)
        print(f'Total: {initial_count}, After dedup: {final_count}, Removed: {initial_count - final_count}')
    
    return df

def get_negativity_score(text, analyzer):
    """
    Calculate negativity using NLTK VADER sentiment analyzer.
    Returns score from 0 (positive) to 1 (negative).
    """
    try:
        text = text[:512]
        scores = analyzer.polarity_scores(text)
        # VADER compound score ranges from -1 (negative) to +1 (positive)
        # We want negativity, so invert and normalize
        negativity = max(0.0, -scores['compound'])
        return negativity
    except:
        return 0.5

def get_cluster_keywords(posts_df, vectorizer, cluster_id, n_keywords=5):
    cluster_posts = posts_df[posts_df['cluster'] == cluster_id]
    cluster_tfidf = vectorizer.transform(cluster_posts['cleaned_content'])
    
    feature_names = vectorizer.get_feature_names_out()
    scores = cluster_tfidf.sum(axis=0).A1
    
    top_indices = scores.argsort()[-n_keywords:][::-1]
    return [feature_names[i] for i in top_indices]

def categorize_cluster(keywords_list):
    keywords_text = ' '.join(keywords_list).lower()
    
    seo_words = ['seo', 'search', 'ranking', 'google', 'keyword', 'optimization']
    geo_words = ['location', 'local', 'geo', 'geographic', 'region', 'area']
    sl_words = ['listening', 'monitoring', 'tracking', 'social', 'brand', 'mention']
    cv_words = ['customer', 'voice', 'feedback', 'review', 'sentiment', 'opinion']
    
    categories = []
    if any(word in keywords_text for word in seo_words):
        categories.append('SEO')
    if any(word in keywords_text for word in geo_words):
        categories.append('GEO')
    if any(word in keywords_text for word in sl_words):
        categories.append('Social Listening')
    if any(word in keywords_text for word in cv_words):
        categories.append('Customer Voice')
    
    return categories or ['General Marketing']

def analyze_issues(time_filter='week', limit=15, num_clusters=8):
    """
    Main analysis pipeline - updated with influence algorithm.
    """
    print(f"Starting analysis (time_filter={time_filter}, limit={limit})")
    
    posts_df = collect_reddit_posts(SUBREDDITS, SEARCH_TERMS, time_filter, limit)
    
    if len(posts_df) == 0:
        return {"error": "No posts collected"}
    
    print(f"Collected {len(posts_df)} posts")
    
    posts_df['full_content'] = (
        posts_df['title'] + ' ' +
        posts_df['text'] + ' ' +
        posts_df['top_comments']
    )
    
    posts_df['cleaned_content'] = posts_df['full_content'].apply(clean_text)
    posts_df = posts_df[posts_df['cleaned_content'].str.len() > 50]
    
    if len(posts_df) < num_clusters:
        num_clusters = max(3, len(posts_df) // 2)
    
    # Calculate influence scores BEFORE clustering
    posts_df['influence'] = posts_df.apply(reddit_influence, axis=1)
    
    vectorizer = TfidfVectorizer(
        max_features=500,
        min_df=2,
        max_df=0.8,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(posts_df['cleaned_content'])
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    posts_df['cluster'] = kmeans.fit_predict(tfidf_matrix)
    
    print("Analyzing sentiment...")
    analyzer = get_sentiment_analyzer()
    posts_df['negativity_score'] = posts_df['cleaned_content'].apply(
        lambda x: get_negativity_score(x, analyzer)
    )
    
    ranked_issues = []
    
    for cluster_id in range(num_clusters):
        cluster_posts = posts_df[posts_df['cluster'] == cluster_id]
        
        if len(cluster_posts) == 0:
            continue
        
        keywords = get_cluster_keywords(posts_df, vectorizer, cluster_id)
        categories = categorize_cluster(keywords)
        
        avg_negativity = cluster_posts['negativity_score'].mean()
        frequency = len(cluster_posts)
        avg_influence = cluster_posts['influence'].mean()
        
        # KEY CHANGE: Use influence-weighted scoring
        final_score = avg_negativity * frequency * (1 + avg_influence/10)
        
        # Sort by influence, not just score
        top_posts = cluster_posts.nlargest(3, 'influence')
        sample_posts = []
        for _, post in top_posts.iterrows():
            sample_posts.append({
                "title": post['title'],
                "url": post['url'],
                "score": int(post['score']),
                "comments": int(post['num_comments']),
                "influence": round(float(post['influence']), 2)
            })
        
        ranked_issues.append({
            'cluster_id': cluster_id,
            'categories': categories,
            'keywords': keywords,
            'frequency': frequency,
            'avg_negativity': round(avg_negativity, 3),
            'avg_influence': round(avg_influence, 2),
            'final_score': round(final_score, 2),
            'sample_posts': sample_posts
        })
    
    ranked_issues = sorted(ranked_issues, key=lambda x: x['final_score'], reverse=True)
    
    for i, issue in enumerate(ranked_issues, 1):
        issue['rank'] = i
    
    print(f"Analysis complete. Found {len(ranked_issues)} issues")
    
    return ranked_issues

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "Feature 4 - SEO/GEO/Social Listening Issues",
        "endpoints": {
            "/": "API info",
            "/health": "Health check",
            "/analyze": "Run issue analysis"
        },
        "note": "Using NLTK VADER for sentiment analysis (lighter, faster)"
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()})

@app.route('/analyze', methods=['GET'])
def analyze():
    try:
        time_filter = request.args.get('time_filter', 'week')
        limit = int(request.args.get('limit', 15))
        
        print(f"Starting analysis (time_filter={time_filter}, limit={limit})")
        
        result = analyze_issues(time_filter, limit)
        
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 400
        
        ranked_complaints = []
        for issue in result:
            ranked_complaints.append({
                "rank": issue['rank'],
                "complaint": {
                    "id": f"issue_{issue['cluster_id']}",
                    "title": ' + '.join(issue['keywords'][:3]),
                    "category": issue['categories'][0] if issue['categories'] else 'General',
                    "subcategories": issue['keywords']
                },
                "metrics": {
                    "frequency": round(issue['frequency'] / sum(i['frequency'] for i in result), 3) if result else 0,
                    "mentions": issue['frequency'],
                    "severity": issue['avg_negativity'],
                    "influence": issue['avg_influence'],
                    "trend_7d": "stable"
                },
                "evidence": [
                    {
                        "title": post["title"],
                        "url": post["url"],
                        "engagement": {
                            "upvotes": post["score"],
                            "comments": post["comments"],
                            "influence": post["influence"]
                        }
                    } for post in issue['sample_posts'][:3]
                ]
            })
        
        response = {
            "data": {
                "ranked_complaints": ranked_complaints
            },
            "meta": {
                "total": len(ranked_complaints),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "time_filter": time_filter,
                "posts_per_search": limit,
                "sentiment_model": "NLTK VADER"
            }
        }
        
        print(f"Analysis complete. Returning {len(ranked_complaints)} issues")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Analysis failed",
            "message": str(e),
            "tip": "Try reducing the 'limit' parameter"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
