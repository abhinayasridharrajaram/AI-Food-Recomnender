import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
from datetime import datetime

import streamlit as st
from streamlit_chat import message
from langchain_community.vectorstores.faiss import FAISS
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
import boto3

from utils import (
    describe_input_image,
    enhance_search,
    clean_text,
    recommend_dishes_by_preference,
    assistant,
)

# -----------------------------
# AWS Bedrock setup
# -----------------------------
BEDROCK_REGION = "us-east-1"
CHAT_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"

bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

embeddings = BedrockEmbeddings(
    client=bedrock,
    model_id=EMBED_MODEL_ID
)

llm = ChatBedrockConverse(
    client=bedrock,
    model=CHAT_MODEL_ID,
    temperature=0.0,
    max_tokens=2048,
)

# -----------------------------
# Load FAISS index
# -----------------------------
db = FAISS.load_local(
    "output/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# -----------------------------
# Helpers
# -----------------------------
def safe_parse_json(raw_response):
    if raw_response is None:
        return {
            "recommendation": "no",
            "response": "I could not generate a response."
        }

    if not isinstance(raw_response, str):
        raw_response = str(raw_response)

    cleaned = raw_response.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned)
    except Exception:
        return {
            "recommendation": "no",
            "response": cleaned if cleaned else "I could not parse the model response."
        }


def ensure_log_dir():
    log_dir = os.path.join("output", "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def serialize_results(results):
    serialized = []
    for doc in results:
        serialized.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })
    return serialized


def write_interaction_log(
    original_input,
    working_input,
    enhanced_search_query,
    image_description,
    raw_chatbot_response,
    parsed_chatbot_response,
    results,
    final_output
):
    log_dir = ensure_log_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "timestamp": timestamp,
        "original_input": original_input,
        "working_input": working_input,
        "enhanced_search_query": enhanced_search_query,
        "image_description": image_description,
        "raw_chatbot_response": raw_chatbot_response,
        "parsed_chatbot_response": parsed_chatbot_response,
        "retrieved_results": serialize_results(results),
        "final_output": final_output
    }

    file_path = os.path.join(log_dir, f"interaction_{timestamp}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return file_path


# -----------------------------
# Session state
# -----------------------------
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "assistant_response" not in st.session_state:
    st.session_state["assistant_response"] = []

# -----------------------------
# UI
# -----------------------------
st.title("Food Recommendation Assistant")

user_input = st.text_input("You:", "", key="input")
original_input = user_input[:]

uploaded_image = st.file_uploader(
    "Upload an image to enhance search",
    type=["png", "jpg", "jpeg"]
)

send_button = st.button("Send")

# -----------------------------
# Main processing
# -----------------------------
if send_button and (user_input.strip() or uploaded_image):
    image_mode = False
    working_input = user_input.strip()
    image_description = ""

    try:
        # Image understanding via direct Bedrock Converse
        if uploaded_image is not None:
            image_mode = True
            try:
                image_description = describe_input_image(
                    uploaded_image=uploaded_image,
                    bedrock_client=bedrock,
                    model_id=CHAT_MODEL_ID
                )

                if working_input:
                    working_input = (
                        f"I am looking for this dish. {working_input}. "
                        f"Image description: {image_description}"
                    )
                else:
                    working_input = (
                        f"I am looking for this dish. Image description: {image_description}"
                    )

                st.caption(f"Image understood as: {image_description}")

            except Exception as e:
                st.warning(f"Image description failed. Continuing with text only. Error: {e}")

        if not working_input:
            st.warning("Please enter a query or upload an image.")
            st.stop()

        st.session_state["past"].append(working_input)

        # Enhance search query
        enhanced_search_query = enhance_search(working_input, llm)
        enhanced_search_query = clean_text(enhanced_search_query)

        # Perform similarity search using enhanced query
        results = db.similarity_search(enhanced_search_query, k=5)
        print("Results generated!")

        # Build context
        context = ""
        for doc in results:
            context += doc.page_content + "\n\n"

        # Assistant response
        raw_chatbot_response = assistant(context, working_input, llm)
        print("RAW CHATBOT RESPONSE:", repr(raw_chatbot_response))

        chatbot_response = safe_parse_json(raw_chatbot_response)

        recommendation = chatbot_response.get("recommendation", "no")
        response = chatbot_response.get("response", "I could not generate a response.")

        st.session_state["assistant_response"].append(response)

        # Recommendation path
        if recommendation == "yes":
            rec_response, relevant_images = recommend_dishes_by_preference(
                results,
                original_input if original_input.strip() else working_input,
                llm
            )

            if not rec_response:
                st.session_state["generated"].append((response, []))
            else:
                st.session_state["generated"].append((rec_response, relevant_images))

            print("Recommendation response generated")
        else:
            st.session_state["generated"].append((response, []))

        final_output = st.session_state["generated"][-1]

        log_file = write_interaction_log(
            original_input=original_input,
            working_input=working_input,
            enhanced_search_query=enhanced_search_query,
            image_description=image_description,
            raw_chatbot_response=raw_chatbot_response,
            parsed_chatbot_response=chatbot_response,
            results=results,
            final_output=final_output
        )

        st.caption(f"Log saved to: {log_file}")

    except Exception as e:
        st.error(f"An error occurred while processing your request: {e}")

# -----------------------------
# Display chat history
# -----------------------------
for i in range(len(st.session_state["generated"]) - 1, -1, -1):
    message(st.session_state["past"][i], is_user=True, key=f"{i}_user")

    response, images = st.session_state["generated"][i]

    if isinstance(response, list):
        image_keys = list(images.keys())

        for j, rec in enumerate(response):
            col1, col2 = st.columns([3, 1])

            metadata = {}
            image_path = None

            if j < len(image_keys):
                image_path = image_keys[j]
                metadata = images.get(image_path, {})

            with col1:
                with st.chat_message("assistant"):
                    st.markdown(f"**{rec}**")
                    st.markdown(f"**Name:** {metadata.get('menu_item_name', 'N/A')}")
                    st.markdown(f"**Restaurant:** {metadata.get('restaurant_name', 'N/A')}")
                    st.markdown(f"**Nutrition:** {metadata.get('nutrition', 'N/A')}")
                    st.markdown(f"**Calories:** {metadata.get('calories', 'N/A')}")
                    st.markdown(f"**Price: USD** {metadata.get('price', 'N/A')}")
                    st.markdown(f"**Serves:** {metadata.get('serves', 'N/A')}")
                    st.markdown(f"**Rating:** {metadata.get('average_rating', 'N/A')}")

            with col2:
                if image_path:
                    full_image_path = os.path.join("data", image_path)
                    if os.path.exists(full_image_path):
                        #st.image(full_image_path, use_container_width=True)
                        st.image(full_image_path, use_column_width=True)
      
            
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**{response}**")