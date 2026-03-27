import re
import string
from langchain_core.messages import HumanMessage, SystemMessage


def extract_text_content(response):
    """
    Normalize model response content into a plain string.
    Works for string responses and list/block-style responses.
    """
    if response is None:
        return ""

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                else:
                    text_parts.append(str(item))
            elif hasattr(item, "text"):
                text_parts.append(item.text)
            else:
                text_parts.append(str(item))
        return " ".join(text_parts).strip()

    return str(content).strip()


def describe_input_image(uploaded_image, bedrock_client, model_id):
    """
    Uses Bedrock Converse directly for image understanding.
    uploaded_image is the Streamlit uploaded file object.
    """
    image_bytes = uploaded_image.getvalue()
    file_name = uploaded_image.name.lower()

    if file_name.endswith(".png"):
        image_format = "png"
    elif file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
        image_format = "jpeg"
    else:
        raise ValueError("Unsupported image format. Please upload PNG, JPG, or JPEG.")

    response = bedrock_client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "Describe this food image briefly for menu search. "
                            "Focus only on the dish, cuisine, visible ingredients, cooking style, "
                            "texture, shape, and useful search keywords. "
                            "Do not mention plate, table, background, utensils, or decoration."
                        )
                    },
                    {
                        "image": {
                            "format": image_format,
                            "source": {
                                "bytes": image_bytes
                            }
                        }
                    }
                ]
            }
        ],
        inferenceConfig={
            "temperature": 0.0,
            "maxTokens": 300
        }
    )

    content_blocks = response["output"]["message"]["content"]
    text_parts = [block["text"] for block in content_blocks if "text" in block]
    return " ".join(text_parts).strip()


def enhance_search(user_input, llm):
    hyde_prompt = [
        SystemMessage(
            content="You are an expert culinary assistant. Your task is to produce a search query description based on user input or preference."
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""You are an expert culinary assistant tasked with generating a search query that helps recommend a variety of menu items based on user preferences.

User Input:
{user_input}

Generate a response that includes only the key unique search terms according to the user's preference. Do not include unnecessary words that do not help search.

The search query may include:
- similar menu items
- cuisines
- key ingredients
- dietary preferences
- nutritional preferences
- allergens to avoid

Keep the output concise and search-friendly."""
                }
            ]
        ),
    ]

    response = llm.invoke(hyde_prompt)
    return extract_text_content(response)


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)

    text = re.sub(r"<.*?>", "", text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()

    return text


def relevance_checker(context, preference, llm):
    relevance_prompt = [
        SystemMessage(
            content="You are a restaurant assistant specializing in helping customers find the food they want."
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""Answer the question "Is this dish relevant to the user by comparing dish details and user preference?" in one word only: Yes or No.

Say Yes only if it is relevant. Otherwise say No.

Context:
{context}

User Preference:
{preference}

Answer:"""
                }
            ]
        ),
    ]

    response = llm.invoke(relevance_prompt)
    return extract_text_content(response)


def dish_summary(dish_description, preference, llm):
    summary_prompt = [
        SystemMessage(
            content="You are a culinary assistant designed to summarize the dish description in accordance with the user preference."
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""Your task is to create a very short two-line summary of the dish in a savory manner by highlighting the user preference.

The summary should explain why the dish is a good fit for the user.
Include dish name, origin, ingredients, and any other relevant information requested by the user.
Do not include unnecessary sentences or extra commentary.

Dish Description:
{dish_description}

User Preference:
{preference}"""
                }
            ]
        ),
    ]

    response = llm.invoke(summary_prompt)
    return extract_text_content(response)


def recommend_dishes_by_preference(search_results, original_input, llm):
    relevant_images = {}
    responses = []

    count = 0
    for doc in search_results:
        relevant = relevance_checker(doc.page_content, original_input, llm)

        if relevant.lower().strip() == "yes":
            image_path = doc.metadata.get("image_path")
            if image_path:
                relevant_images[image_path] = doc.metadata

            responses.append(dish_summary(doc.page_content, original_input, llm))
            count += 1

        if count == 3:
            break

    return responses, relevant_images


def assistant(context, user_input, llm):
    assistant_prompt = [
        SystemMessage(
            content="You are a helpful and knowledgeable assistant capable of providing food recommendations and answering general queries."
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""Your task is to engage users in natural, friendly dialogue to understand their preferences, dietary restrictions, and culinary interests.

Your goal is to summarize relevant food recommendations in a single sentence if the user query indicates that they want a recommendation.
Otherwise, ask the user to provide preferences such as cuisine, dish type, dietary needs, or flavor profile.
Do not answer if you do not have relevant knowledge about the query.

Remember the context given is all the dishes we have.

User Input:
{user_input}

Context:
{context}

Return ONLY valid JSON. Do not include markdown, code fences, or extra commentary.

Use exactly this structure:
{{
  "recommendation": "yes" or "no",
  "response": "your response here"
}}"""
                }
            ]
        ),
    ]

    response = llm.invoke(assistant_prompt)
    return extract_text_content(response)