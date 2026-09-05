import os
from dotenv import load_dotenv
import streamlit as st
from google import genai

# ----------------------------
# Load API Key
# ----------------------------
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env file")
    st.stop()

client = genai.Client(api_key=API_KEY)

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Restaurant Review Response Generator",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ Restaurant Review Response Generator")
st.write("Generate professional responses to customer reviews using Gemini AI.")

st.divider()

# ----------------------------
# Inputs
# ----------------------------

rating = st.selectbox(
    "⭐ Select Star Rating",
    [1, 2, 3, 4, 5]
)

review = st.text_area(
    "Customer Review",
    placeholder="Type the customer's review here..."
)

# ----------------------------
# Generate Reply Function
# ----------------------------

def generate_reply(review, rating):

    if rating == 1 and review.strip() == "":
        review = "No reason provided."

    prompt = f"""
You are a professional restaurant manager.

Generate a polite and professional reply.

Rules:

5 Stars:
Thank the customer warmly and invite them back.

4 Stars:
Thank the customer and appreciate the feedback.

3 Stars:
Thank them and mention improvements.

2 Stars:
Apologize for not meeting expectations and promise improvements.

1 Star:
Offer a sincere apology.
Show empathy.
Invite the customer to contact the restaurant.
Promise corrective action.

Reply should be under 100 words.

Do NOT use placeholders like [Phone Number] or [Email].

Star Rating:
{rating}

Customer Review:
{review}
"""

    chat = client.chats.create(
        model="gemini-3.6-flash"
    )

    response = chat.send_message(prompt)

    return response.text

# ----------------------------
# Button
# ----------------------------

if st.button("Generate Reply", use_container_width=True):

    if review.strip() == "" and rating != 1:
        st.warning("Please enter a customer review.")
    else:
        try:

            with st.spinner("Generating response..."):

                reply = generate_reply(review, rating)

            st.success("Reply Generated Successfully!")

            st.subheader("Restaurant Reply")

            st.write(reply)

        except Exception as e:

            st.error(f"Error : {e}")

st.divider()

st.caption("Powered by Gemini 3.6 Flash")