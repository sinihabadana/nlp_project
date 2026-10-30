import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

st.set_page_config(
    page_title="ReviewLens | Electronics Review Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}
.stApp{background:linear-gradient(180deg,#f7f9fc 0%,#eef3f9 100%)}
.block-container{max-width:1280px;padding:1.6rem 2rem 3rem}
.hero{border-radius:26px;padding:2.25rem 2.5rem;color:white;background:linear-gradient(135deg,#111827,#244a7c 60%,#2f6598);box-shadow:0 18px 45px rgba(15,23,42,.16);margin-bottom:1.2rem}
.hero .pill{display:inline-block;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.16);padding:.4rem .8rem;border-radius:999px;font-size:.76rem;font-weight:800;letter-spacing:.02em}
.hero h1{font-size:2.45rem;line-height:1.1;margin:.85rem 0 .5rem;font-weight:800;letter-spacing:-.04em}
.hero p{margin:0;color:#dbeafe;font-size:1rem}
.section{font-size:1.15rem;font-weight:800;color:#0f172a;margin:1.25rem 0 .65rem}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:1rem 1.1rem;box-shadow:0 8px 24px rgba(15,23,42,.05)}
.metric{min-height:96px}.metric-label{font-size:.76rem;color:#64748b;font-weight:700}.metric-value{font-size:1.55rem;color:#0f172a;font-weight:800;margin-top:.25rem}
.rec{padding:1.25rem 1.4rem;border-radius:20px;background:linear-gradient(135deg,#eff6ff,#ffffff);border:1px solid #bfdbfe;box-shadow:0 8px 24px rgba(37,99,235,.07)}
.rec h3{margin:0;color:#0f172a}.rec p{margin:.4rem 0 0;color:#475569;line-height:1.55}
.theme{display:inline-block;background:#f1f5f9;border:1px solid #e2e8f0;color:#334155;padding:.38rem .62rem;border-radius:999px;margin:.2rem;font-size:.78rem}
.smallnote{font-size:.78rem;color:#64748b}
.product-search{background:white;border:1px solid #dbe3ef;border-radius:18px;padding:1rem}

div[data-testid="stSidebar"]{background:#0f172a}
div[data-testid="stSidebar"] *{color:#e2e8f0}
div[data-testid="stSidebar"] .stSelectbox label{color:#cbd5e1}
.stButton>button{border-radius:12px;font-weight:800}
</style>
""", unsafe_allow_html=True)

DATA_PATH = "data/DatafinitiElectronicsProductData.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, low_memory=False)

    data["reviews.rating"] = pd.to_numeric(
        data["reviews.rating"], errors="coerce"
    )
    data["reviews.text"] = data["reviews.text"].fillna("").astype(str)
    data["reviews.title"] = data["reviews.title"].fillna("").astype(str)
    data["name"] = data["name"].fillna("Unknown Product").astype(str)
    data["asins"] = data["asins"].fillna("").astype(str)
    data["brand"] = data["brand"].fillna("").astype(str)

    data = data.dropna(subset=["reviews.rating"])
    data = data[data["reviews.rating"].between(1, 5)]
    data = data[data["reviews.text"].str.strip().ne("")].copy()

    # NLTK stop words
    stop_words = set(stopwords.words("english"))
    stop_words = stop_words - {"no", "nor", "not"}

    def clean(text):
        text = str(text).lower()

        text = re.sub(
            r"<[^>]+>|https?://\S+|www\.\S+",
            " ",
            text
        )
        text = re.sub(r"[^a-z\s]", " ", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Remove stop words using NLTK
        words = text.split()
        words = [word for word in words if word not in stop_words]

        return " ".join(words)

    data["clean_text"] = data["reviews.text"].map(clean)

    data = data[data["clean_text"].str.len() > 2].copy()

    data["rating_sentiment"] = np.select(
        [
            data["reviews.rating"] <= 2,
            data["reviews.rating"] == 3,
            data["reviews.rating"] >= 4
        ],
        ["Negative", "Neutral", "Positive"],
        default="Neutral"
    )

    return data.reset_index(drop=True)

@st.cache_resource
def train_model(texts, labels):

    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2,
        max_features=12000,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=.20, random_state=42, stratify=labels
    )

    model = LogisticRegression(
        max_iter=1200, 
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)


    acc = accuracy_score(y_test, pred)
    p,r,f,_ = precision_recall_fscore_support(
        y_test,pred,
        average="weighted",
        zero_division=0
    )
    return vectorizer, model, {
        "accuracy":acc,
        "precision":p,
        "recall":r,
        "f1":f
    }

def top_terms(texts, n=8):
    texts=[str(x) for x in texts if str(x).strip()]
    if not texts: return []
    v=TfidfVectorizer(
        ngram_range=(1,2),
        min_df=1,
        max_features=2000
    )
    X=v.fit_transform(texts)
    scores=np.asarray(X.mean(axis=0)).ravel()
    terms=np.array(v.get_feature_names_out())
    ids=scores.argsort()[::-1]
    return [terms[i] for i in ids if scores[i]>0][:n]

def recommendation(avg,pos,neg):
    if avg>=4 and pos>=70 and neg<=15:
        return (
            "RECOMMENDED",
            "The product has a high average rating and "
            "a strongly positive review distribution."
        )
    if avg>=3 and pos>=50 and neg<30:
        return (
            "CONSIDER",
            "The feedback is reasonably positive, but "
            "there are some issues worth reviewing."
        )
    return (
        "NOT RECOMMENDED",
        "The aggregate rating/sentiment indicates a "
        "comparatively high level of dissatisfaction."
    )

def nltk_tokenize_review(text):
    """Tokenize review text using NLTK."""
    return wordpunct_tokenize(str(text))

def nltk_stem_review(text):
    stemmer = PorterStemmer()
    tokens = wordpunct_tokenize(str(text).lower())
    return [stemmer.stem(word) for word in tokens if word.isalpha()]

df=load_data()

st.markdown("""
<div class="hero">
<div class="pill">PRODUCT REVIEW ANALYSIS SYSTEM</div>
<h1>ReviewLens</h1>
<p>Amazon & Best Buy Electronics — product-level review intelligence and purchase analysis.</p>
</div>
""",unsafe_allow_html=True)

# Dataset facts from the actual uploaded file
total_reviews=len(df)
total_products=df["asins"].nunique()
total_ratings=len(df)
missing_original=164  # from the supplied raw dataset; shown only in dataset note below

with st.sidebar:
    st.markdown("## 🔎 ReviewLens")
    st.caption("Analyze all available reviews for one electronic product.")
    st.divider()
    st.markdown("### Dataset")
    st.write(f"**Usable reviews:** {total_reviews:,}")
    st.write(f"**Products:** {total_products}")
    st.write("**Original rows:** 7,299")
    st.caption("Rows without a usable rating/review are excluded from NLP analysis.")

# Product search + selector are intentionally placed in the main page so they are always visible.
st.markdown('<div class="section">Choose a Product</div>',unsafe_allow_html=True)
search_product=st.text_input(
    "Search product name",
    placeholder="Type part of a product name, e.g. Logitech, Roku, Fire TV...",
    label_visibility="collapsed"
)
products=(df.groupby(["asins","name"]).size()
          .reset_index(name="review_count")
          .sort_values(["name"],ascending=[True])
          .reset_index(drop=True))

if search_product.strip():
    filtered_products=products[products["name"].str.contains(search_product.strip(),case=False,na=False)].copy()
else:
    filtered_products=products.copy()

if filtered_products.empty:
    st.warning("No matching product found. Try a shorter product name or clear the search box.")
    st.stop()

product_options=[f"{r['name']}  •  {int(r['review_count'])} reviews" for _,r in filtered_products.iterrows()]
selected_option=st.selectbox(
    f"Select from {len(filtered_products)} matching product(s)",
    product_options,
    help="The dataset contains 50 distinct ASIN/product IDs. Search first or browse the dropdown."
)
selected_row=filtered_products.iloc[product_options.index(selected_option)]
selected_asin=selected_row["asins"]
selected_name=selected_row["name"]

st.caption(f"Showing **{len(filtered_products)}** matching products out of **{total_products}** products in the dataset.")

product=df[df["asins"]==selected_asin].copy()
vectorizer,model,model_metrics=train_model(tuple(df["clean_text"]),tuple(df["rating_sentiment"]))

Xp=vectorizer.transform(product["clean_text"])
product["sentiment"]=model.predict(Xp)
product["confidence"]=model.predict_proba(Xp).max(axis=1)*100

avg=float(
    product["reviews.rating"].mean()
)
pos=float(
    (product.sentiment=="Positive").mean()*100
)
neu=float(
    (product.sentiment=="Neutral").mean()*100
)
neg=float(
    (product.sentiment=="Negative").mean()*100
)
rec,detail=recommendation(avg,pos,neg)

brand=product["brand"].mode().iat[0] if not product["brand"].mode().empty else "Electronics"

st.markdown('<div class="section">Product Overview</div>',unsafe_allow_html=True)
st.markdown(f"""
<div class="card">
<h3 style="margin:0;color:#0f172a">{selected_name}</h3>
<p style="margin:.35rem 0 0;color:#64748b"><b>{brand}</b> &nbsp;•&nbsp; ASIN: {selected_asin}</p>
</div>
""",unsafe_allow_html=True)

metric_cards=[
("Reviews analyzed",f"{len(product):,}"),
("Average rating",f"{avg:.2f} / 5"),
("Positive",f"{pos:.1f}%"),
("Neutral",f"{neu:.1f}%"),
("Negative",f"{neg:.1f}%")
]
cols=st.columns(5)
for c,(lab,val) in zip(cols,metric_cards):
    with c:
        st.markdown(f'<div class="card metric"><div class="metric-label">{lab}</div><div class="metric-value">{val}</div></div>',unsafe_allow_html=True)

st.markdown('<div class="section">Customer Opinion</div>',unsafe_allow_html=True)
c1,c2=st.columns(2)
with c1:
    s=pd.DataFrame({
        "Sentiment":["Positive","Neutral","Negative"],
        "Percentage":[pos,neu,neg]
    })
    fig=px.pie(
        s,
        names="Sentiment",
        values="Percentage",
        hole=.58,
        title="Sentiment Distribution"
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10,r=10,t=55,b=10),
        legend_title=""
    )
    st.plotly_chart(fig,use_container_width=True)
with c2:
    r=(
        product["reviews.rating"]
        .round()
        .astype(int)
        .value_counts()
        .reindex([1,2,3,4,5],fill_value=0)
        .reset_index()
    )
    r.columns=["Rating","Reviews"]
    fig=px.bar(
        r,
        x="Rating",
        y="Reviews",
        text="Reviews",
        title="Rating Distribution"
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10,r=10,t=55,b=10),
        xaxis=dict(dtick=1),
        margin_autoexpand=True
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig,use_container_width=True)

positive_terms=top_terms(product.loc[product.sentiment=="Positive","reviews.text"])
negative_terms=top_terms(product.loc[product.sentiment=="Negative","reviews.text"])

c1,c2=st.columns(2)
with c1:
    html="".join(f'<span class="theme">✓ {x}</span>' for x in positive_terms) or "No strong positive themes detected."
    st.markdown(f'<div class="card"><h4 style="margin:0 0 .45rem">👍 What customers like</h4>{html}</div>',unsafe_allow_html=True)
with c2:
    html="".join(f'<span class="theme">⚠ {x}</span>' for x in negative_terms) or "No strong complaint themes detected."
    st.markdown(f'<div class="card"><h4 style="margin:0 0 .45rem">⚠ Common complaints</h4>{html}</div>',unsafe_allow_html=True)

st.markdown(f"""
<div class="rec">
<h3>Purchase Analysis: {rec}</h3>
<p>{detail}</p>
<p class="smallnote"><b>How it is calculated:</b> average rating + model-predicted sentiment percentages. This is a transparent project rule, not a guarantee of product quality.</p>
</div>
""",unsafe_allow_html=True)

st.markdown('<div class="section">Review Explorer</div>',unsafe_allow_html=True)
f1,f2,f3=st.columns([1,1,1.5])
with f1: sentiment_filter=st.selectbox("Sentiment",["All","Positive","Neutral","Negative"])
with f2: rating_filter=st.selectbox("Rating",["All",1,2,3,4,5])
with f3: search=st.text_input("Search reviews",placeholder="Try: battery, quality, sound...")

view=product.copy()
if sentiment_filter!="All": view=view[view.sentiment==sentiment_filter]
if rating_filter!="All": view=view[view["reviews.rating"].astype(int)==int(rating_filter)]
if search.strip(): view=view[view["reviews.text"].str.contains(search.strip(),case=False,na=False)]

display=view[["reviews.rating","sentiment","reviews.title","reviews.text","confidence"]].copy()
display.columns=["Rating","Sentiment","Review Title","Review","Model Confidence"]
display["Rating"]=display["Rating"].map(lambda x:f"{x:.0f} ⭐")
display["Model Confidence"]=display["Model Confidence"].map(lambda x:f"{x:.1f}%")
st.caption(f"Showing {min(len(display),100):,} of {len(display):,} matching reviews.")
st.dataframe(display.head(100),use_container_width=True,hide_index=True)

with st.expander("📌 Model & dataset details"):
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Accuracy",f"{model_metrics['accuracy']*100:.2f}%")
    m2.metric("Precision",f"{model_metrics['precision']*100:.2f}%")
    m3.metric("Recall",f"{model_metrics['recall']*100:.2f}%")
    m4.metric("F1-score",f"{model_metrics['f1']*100:.2f}%")
    st.markdown("""
**Sentiment labels used for training:** 1–2 stars = Negative, 3 stars = Neutral, 4–5 stars = Positive.

**NLP:** review cleaning → TF-IDF unigram/bigram features → Logistic Regression.

**Important:** the dataset does not contain a separate sentiment column. Sentiment labels are derived from the star ratings for this academic project.

**Source file used by this application:** `DatafinitiElectronicsProductData.csv`.
""")

st.markdown('<div class="smallnote">ReviewLens • Python • Pandas • Scikit-learn • Plotly • Streamlit</div>',unsafe_allow_html=True)


