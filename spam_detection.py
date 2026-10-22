# Mini Project - Email Spam Detection (Web App version)
# Riyanshi Biyani, 25I5096
# same TF-IDF + Naive Bayes model as before, just added a simple browser UI using streamlit

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon=":material/mark_email_unread:",
    layout="centered",
)

# ---------------- dataset (same one i used before) ----------------
raw_data = [
    ("Congratulations! You have won a $1000 Walmart gift card. Click here to claim now", "spam"),
    ("Win a brand new iPhone! Text WIN to 80086 now, limited offer", "spam"),
    ("URGENT! Your account has been suspended. Verify your bank details immediately", "spam"),
    ("You have been selected for a free cruise to the Bahamas, call now", "spam"),
    ("Get cheap loans with 0% interest, apply today, no credit check", "spam"),
    ("FREE entry into our $5000 prize draw, text WIN now, offer ends soon", "spam"),
    ("Claim your free lottery winnings of $10,000 now, reply with your bank info", "spam"),
    ("Hot singles in your area want to meet you tonight, click link", "spam"),
    ("Your Netflix subscription payment failed, update card details here", "spam"),
    ("Limited time offer, buy one get one free on all medicines, order now", "spam"),
    ("Congratulations, you've been chosen to receive a free laptop, click to claim", "spam"),
    ("Earn $5000 per week working from home, no experience needed, sign up now", "spam"),
    ("Your package could not be delivered, click here to reschedule and pay fee", "spam"),
    ("You are pre-approved for a credit card with a $10,000 limit, apply now", "spam"),
    ("Act now! Your PayPal account will be locked, confirm your identity", "spam"),
    ("Get 90% discount on all branded watches only today, buy now", "spam"),
    ("You have an unclaimed inheritance of $2 million, contact us urgently", "spam"),
    ("Double your bitcoin investment in 24 hours, guaranteed returns", "spam"),
    ("Your resume has been shortlisted, join the interview using this link", "spam"),
    ("Please pay $499 immediately to avoid legal action, click link", "spam"),
    ("You've won a free vacation package to Hawaii, call now to claim", "spam"),
    ("Last chance! 70% off everything, sale ends at midnight, shop now", "spam"),
    ("Verify your account now or it will be permanently deleted", "spam"),
    ("Make money fast! Earn $500 a day working online, no skills required", "spam"),
    ("Your card has been charged $299, click here if this wasn't you", "spam"),
    ("Free gift waiting for you, just pay shipping and handling", "spam"),
    ("You are the lucky winner of our monthly draw, claim your prize now", "spam"),
    ("Exclusive deal just for you, click to unlock 80% discount today", "spam"),
    ("Your computer has a virus, download this software immediately to fix it", "spam"),
    ("Congratulations! You qualify for a government grant, apply within 24 hours", "spam"),
    ("Singles near you are waiting, sign up free and start chatting now", "spam"),
    ("Refinance your home loan today and save thousands, apply now", "spam"),
    ("Your SIM card will be blocked, update your KYC details urgently", "spam"),
    ("You have 1 new voicemail, click to listen using this link", "spam"),
    ("Get rich quick with this one simple trick, banks hate it", "spam"),
    ("Reminder: Your subscription will renew tomorrow at 9 AM", "ham"),
    ("Hi John, can we reschedule our meeting to 3 PM tomorrow?", "ham"),
    ("Please find attached the quarterly sales report for review", "ham"),
    ("Mom, I'll be home late tonight, don't wait for dinner", "ham"),
    ("Your Amazon order has been shipped and will arrive on Thursday", "ham"),
    ("Team, the project deadline has been moved to next Friday", "ham"),
    ("Thanks for the great presentation today, well done!", "ham"),
    ("Can you send me the notes from yesterday's class?", "ham"),
    ("Let's catch up for coffee this weekend, are you free Saturday?", "ham"),
    ("Your electricity bill for this month is ready, view it online", "ham"),
    ("The doctor's appointment is confirmed for 10 AM on Monday", "ham"),
    ("Happy birthday! Hope you have a wonderful day", "ham"),
    ("Please review and sign the attached contract by Friday", "ham"),
    ("Reminder: Submit your assignment before midnight", "ham"),
    ("The flight booking confirmation and itinerary are attached", "ham"),
    ("Congratulations on your promotion, well deserved!", "ham"),
    ("Can we discuss the budget numbers in tomorrow's call?", "ham"),
    ("Your OTP for login is 349201, valid for 10 minutes", "ham"),
    ("Lunch at the usual place tomorrow? Let me know", "ham"),
    ("The server maintenance is scheduled for Sunday night", "ham"),
    ("Meeting is rescheduled to 4 PM, please update your calendar", "ham"),
    ("Can you pick up milk on your way home?", "ham"),
    ("The invoice for last month's services is attached for your records", "ham"),
    ("Great job on closing the deal today, the client was impressed", "ham"),
    ("Reminder: Parent-teacher meeting is scheduled for Thursday at 5 PM", "ham"),
    ("I'll send over the revised draft by end of day", "ham"),
    ("Happy anniversary! Wishing you both many more years of happiness", "ham"),
    ("The train is running 15 minutes late, see you at the station", "ham"),
    ("Please review the attached minutes from today's meeting", "ham"),
    ("Let's plan the weekend trip, what dates work for everyone?", "ham"),
    ("Your library book is due for return next Monday", "ham"),
    ("Thanks for your help with the move, really appreciate it", "ham"),
    ("The gym class timing has changed to 7 AM starting next week", "ham"),
    ("Can you review my code and share feedback before the demo?", "ham"),
    ("Dinner reservation confirmed for 8 PM on Saturday", "ham"),
    ("The report looks good, just fix the formatting on page 3", "ham"),
    ("See you at the conference next week, safe travels", "ham"),
    ("Your salary has been credited to your account", "ham"),
    ("Please bring your laptop charger for tomorrow's workshop", "ham"),
]

df = pd.DataFrame(raw_data, columns=["text", "label"])

# training the model - streamlit reruns this file on every interaction,
# so caching it makes sure we don't retrain again and again
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    tfidf = TfidfVectorizer(stop_words="english")
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)

    y_pred = nb_model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["spam", "ham"])
    report = classification_report(y_test, y_pred)

    return tfidf, nb_model, acc, cm, report

tfidf, nb_model, acc, cm, report = train_model()

# ---------------- UI part ----------------
def load_example(message):
    st.session_state["message_input"] = message


def clear_check_history():
    st.session_state["check_history"] = []


st.session_state.setdefault("check_history", [])

st.title("Email spam detector", icon=":material/mark_email_unread:")
st.caption("Mini project by Riyanshi Biyani (25I5096) — IET, DAVV")
st.write("Check a message with a TF-IDF + Naive Bayes model.")

st.subheader("Sample messages")
st.caption("Browse all labeled examples in the demo dataset and load one into the checker.")
spam_examples = df.loc[df["label"] == "spam", "text"].tolist()
genuine_examples = df.loc[df["label"] == "ham", "text"].tolist()
spam_col, genuine_col = st.columns(2)
with spam_col:
    with st.container(border=True):
        st.markdown(f"**Spam examples · {len(spam_examples)} messages**")
        selected_spam = st.selectbox(
            "Choose a spam message",
            spam_examples,
            key="selected_spam_example",
            format_func=lambda message: (
                message if len(message) <= 70 else f"{message[:67]}..."
            ),
        )
        st.write(selected_spam)
        st.button(
            "Load spam message",
            key="load_spam_example",
            on_click=load_example,
            args=(selected_spam,),
            icon=":material/search:",
            width="stretch",
        )

with genuine_col:
    with st.container(border=True):
        st.markdown(f"**Genuine examples · {len(genuine_examples)} messages**")
        selected_genuine = st.selectbox(
            "Choose a genuine message",
            genuine_examples,
            key="selected_genuine_example",
            format_func=lambda message: (
                message if len(message) <= 70 else f"{message[:67]}..."
            ),
        )
        st.write(selected_genuine)
        st.button(
            "Load genuine message",
            key="load_genuine_example",
            on_click=load_example,
            args=(selected_genuine,),
            icon=":material/search:",
            width="stretch",
        )

with st.form("message_check", border=True):
    user_input = st.text_area(
        "Message to check",
        height=140,
        key="message_input",
        placeholder="For example: Congratulations! You won a free iPhone. Click here to claim.",
    )
    submitted = st.form_submit_button(
        "Check message",
        type="primary",
        icon=":material/search:",
        width="stretch",
    )

if submitted:
    if not user_input.strip():
        st.warning("Please enter a message before checking it.")
    else:
        vec = tfidf.transform([user_input])
        pred = nb_model.predict(vec)[0]
        probabilities = dict(zip(nb_model.classes_, nb_model.predict_proba(vec)[0]))
        spam_score = probabilities["spam"] * 100
        ham_score = probabilities["ham"] * 100
        st.session_state["check_history"].insert(
            0,
            {
                "Checked at": pd.Timestamp.now().strftime("%H:%M:%S"),
                "Result": "Spam" if pred == "spam" else "Genuine",
                "Spam score": f"{spam_score:.1f}%",
                "Genuine score": f"{ham_score:.1f}%",
                "Message": user_input.strip(),
            },
        )
        del st.session_state["check_history"][20:]

        with st.container(border=True):
            if pred == "spam":
                st.error("This message looks like spam.", icon=":material/warning:")
            else:
                st.success("This message looks genuine.", icon=":material/check_circle:")

            score_col1, score_col2 = st.columns(2)
            score_col1.metric("Spam score", f"{spam_score:.1f}%", border=True)
            score_col2.metric("Genuine score", f"{ham_score:.1f}%", border=True)
            st.progress(
                spam_score / 100,
                text=f"Estimated spam score: {spam_score:.1f}%",
            )
            st.caption(
                "Scores are estimates from this demo model, not a guarantee that a message is safe."
            )

st.subheader("Recent checks")
st.caption("Up to 20 checks are kept for this session only.")
if st.session_state["check_history"]:
    st.button(
        "Clear history",
        key="clear_check_history",
        on_click=clear_check_history,
        icon=":material/delete:",
    )
    st.dataframe(
        pd.DataFrame(st.session_state["check_history"]),
        hide_index=True,
        alt="Recent message checks and their spam predictions",
    )
else:
    st.caption("Your checked messages will appear here.")

st.subheader("Model performance")
with st.container(horizontal=True):
    st.metric("Test accuracy", f"{acc:.1%}", border=True)
    st.metric("Labeled messages", f"{len(df)}", border=True)
    st.metric("Spam examples", f"{(df['label'] == 'spam').sum()}", border=True)
    st.metric("Genuine examples", f"{(df['label'] == 'ham').sum()}", border=True)

st.warning(
    "This project uses a small set of sample messages. Its test score may not reflect "
    "how well it works on real-world emails or SMS."
)

with st.expander("View evaluation details"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("Confusion matrix (rows: actual, columns: predicted)")
        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual spam", "Actual genuine"],
                columns=["Predicted spam", "Predicted genuine"],
            ),
            alt="Confusion matrix for the spam detector",
        )
    with col2:
        st.write("Classification report")
        st.text(report)