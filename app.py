# app.py
# CV চ্যাটবট (RAG): CV টেক্সট ফাইল থেকে প্রাসঙ্গিক তথ্য খুঁজে LLM দিয়ে উত্তর তৈরি করে।
# চালানো: streamlit run app.py

import streamlit as st                                          # ওয়েব পেজ বানানোর লাইব্রেরি
import requests                                                 # Ollama (local মডেল) এর সাথে কথা বলার জন্য
from sklearn.feature_extraction.text import TfidfVectorizer     # লেখাকে সংখ্যার vector বানায়
from sklearn.metrics.pairwise import cosine_similarity          # দুটো vector কতটা মেলে তা মাপে


# ---------- ধাপ ১: Chunking (টেক্সট ভাগ করা) ----------
def split_into_chunks(text):
    # ফাঁকা লাইন (\n\n) দিয়ে প্যারাগ্রাফে ভাগ করা। প্রতিটি প্যারাগ্রাফ একটি চাঙ্ক।
    parts = text.split("\n\n")
    # খালি অংশ বাদ দিয়ে শুধু কাজের চাঙ্ক রাখা
    return [p.strip() for p in parts if p.strip()]


# ---------- ধাপ ২: Retrieval (প্রাসঙ্গিক অংশ খোঁজা) ----------
def retrieve(question, chunks, top_k=3):
    # token_pattern: লেখাকে শব্দে ভাগ করার নিয়ম। স্পেস ও বাংলা/ইংরেজি বিরামচিহ্নে ভাগ হবে।
    vectorizer = TfidfVectorizer(token_pattern=r"[^\s।,.?!:;()\-]+")
    chunk_vectors = vectorizer.fit_transform(chunks)            # সব চাঙ্ককে vector বানানো
    question_vector = vectorizer.transform([question])          # প্রশ্নকেও একই নিয়মে vector বানানো
    scores = cosine_similarity(question_vector, chunk_vectors)[0]   # প্রতিটি চাঙ্কের মিলের স্কোর (০ থেকে ১)
    best_indexes = scores.argsort()[::-1][:top_k]               # সবচেয়ে বেশি স্কোরের top_k চাঙ্কের নম্বর
    # স্কোর ০ মানে কোনো শব্দ মেলেনি, তাই সেগুলো বাদ দেওয়া
    return [chunks[i] for i in best_indexes if scores[i] > 0]


# ---------- ধাপ ৩: Generation (উত্তর লেখা) ----------
def build_prompt(question, context_chunks):
    # মডেলকে নির্দেশ দেওয়া: শুধু দেওয়া CV তথ্য থেকে উত্তর দাও, নিজে কিছু বানাবে না
    context = "\n\n".join(context_chunks)
    return (
        "তুমি একজন সহকারী। নিচের CV তথ্য ব্যবহার করে প্রশ্নের উত্তর দাও।\n"
        "তথ্য না থাকলে লেখো 'CV-তে এই তথ্য নেই'। নিজে থেকে কিছু বানাবে না।\n\n"
        f"CV তথ্য:\n{context}\n\n"
        f"প্রশ্ন: {question}\n"
        "উত্তর:"
    )


def ask_local(prompt):
    # Ollama অবশ্যই চালু থাকতে হবে (ডিফল্ট পোর্ট 11434)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False},  # আপনি যে মডেল pull করেছেন তার নাম
        timeout=180,                                            # লোকাল মডেল ধীর হলে ১৮০ সেকেন্ড অপেক্ষা
    )
    return response.json()["response"]                          # উত্তরের লেখাটা নেওয়া


def ask_cloud(prompt, api_key):
    from openai import OpenAI                                   # cloud বেছে নিলেই লাইব্রেরি লোড হবে
    client = OpenAI(api_key=api_key)                            # আপনার API key দিয়ে client তৈরি
    response = client.chat.completions.create(
        model="gpt-4o-mini",                                    # সস্তা ও দ্রুত মডেল
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content                  # উত্তরের লেখা


# ---------- Streamlit UI ----------
st.set_page_config(page_title="CV চ্যাটবট", layout="centered")
st.title("আমার CV চ্যাটবট")

# সেশনে চ্যাটের ইতিহাস রাখা। ব্রাউজার ট্যাব বন্ধ করলে ইতিহাস মুছে যাবে।
if "history" not in st.session_state:
    st.session_state.history = []

# বাঁ পাশের sidebar: মডেল ও API key সেটিংস
with st.sidebar:
    st.header("সেটিংস")
    model_choice = st.radio("মডেল বেছে নিন", ["Local (Ollama)", "Cloud (OpenAI)"])
    api_key = ""
    if model_choice == "Cloud (OpenAI)":
        api_key = st.text_input("OpenAI API key", type="password")

# CV ফাইল আপলোড
uploaded = st.file_uploader("আপনার CV ফাইল আপলোড করুন (.txt, UTF-8)", type=["txt"])

if uploaded is None:
    st.info("প্রথমে একটি .txt CV ফাইল আপলোড করুন।")
    st.stop()                                                   # ফাইল না থাকলে নিচের কোড চলবে না

# আপলোড করা ফাইলের বাইট থেকে টেক্সট বানানো
cv_text = uploaded.getvalue().decode("utf-8")
chunks = split_into_chunks(cv_text)

if not chunks:
    st.error("CV ফাইলটি খালি। তথ্য লিখে আবার আপলোড করুন।")
    st.stop()

# আগের চ্যাট বার্তাগুলো দেখানো
for role, message in st.session_state.history:
    with st.chat_message(role):
        st.write(message)

# নতুন প্রশ্ন নেওয়া
question = st.chat_input("CV সম্পর্কে প্রশ্ন করুন...")

if question:
    # প্রশ্নটি স্ক্রিনে দেখানো ও ইতিহাসে সেভ করা
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("উত্তর তৈরি হচ্ছে..."):
            relevant = retrieve(question, chunks)               # ধাপ ২: প্রাসঙ্গিক চাঙ্ক খোঁজা

            if not relevant:
                answer = "CV-তে এই বিষয়ে কোনো তথ্য পাওয়া যায়নি।"
            else:
                prompt = build_prompt(question, relevant)       # ধাপ ৩: prompt বানানো
                try:
                    if model_choice == "Local (Ollama)":
                        answer = ask_local(prompt)              # লোকাল মডেলে উত্তর
                    elif api_key:
                        answer = ask_cloud(prompt, api_key)     # cloud মডেলে উত্তর
                    else:
                        answer = "sidebar-এ OpenAI API key দিন।"
                except Exception as e:
                    answer = f"ত্রুটি হয়েছে: {e}"

        st.write(answer)

    st.session_state.history.append(("assistant", answer))
