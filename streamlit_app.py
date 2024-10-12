# import streamlit as st
# from copilot import Copilot
# import os
# ### set openai key, first check if it is in environment variable, if not, check if it is in streamlit secrets, if not, raise error


# st.title("Chat with an ETF expert")
# st.write(
#     "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
# )

# openai_api_key = os.getenv("OPENAI_API_KEY")

# if not openai_api_key: ### get openai key from user input
#     openai_api_key = st.text_input("Please enter your OpenAI API Key", type="password")

# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.", icon="🗝️")
# else:
#     if "messages" not in st.session_state.keys():  # Initialize the chat messages history
#         st.session_state.messages = [
#             {"role": "assistant", "content": "I am an expert in ETFs. You can ask me any question you want to know about ETFs, and I will try my best to answer them."}
#         ]

#     @st.cache_resource
#     def load_copilot():
#         return Copilot(key = openai_api_key)



#     if "chat_copilot" not in st.session_state.keys():  # Initialize the chat engine
#         st.session_state.chat_copilot = load_copilot()

#     if prompt := st.chat_input(
#         "Ask a question"
#     ):  # Prompt for user input and save to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})

#     for message in st.session_state.messages:  # Write message history to UI
#         with st.chat_message(message["role"]):
#             st.write(message["content"])

#     # If last message is not from assistant, generate a new response
#     if st.session_state.messages[-1]["role"] != "assistant":
#         with st.chat_message("assistant"):

#             retrived_info, answer = st.session_state.chat_copilot.ask(prompt, messages=st.session_state.messages[:-1])
#             ### answer can be a generator or a string

#             #print(retrived_info)
#             if isinstance(answer, str):
#                 st.write(answer)
#             else:
#                 ### write stream answer to UI
#                 def generate():
#                     for chunk in answer:
#                         content = chunk.choices[0].delta.content
#                         if content:
#                             yield content
#                 answer = st.write_stream(generate())

#             st.session_state.messages.append({"role": "assistant", "content": answer})



# import streamlit as st
# from copilot import Copilot
# import os

# # Set the page configuration
# st.set_page_config(
#     page_title="ETF Expert Chat",
#     page_icon="💬",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # Title and instructions
# st.title("💬 Chat with an ETF Expert")
# st.markdown(
#     """
#     Welcome to the ETF expert chatbot! Ask any questions you have about Exchange-Traded Funds (ETFs) and get real-time insights.
#     ### Instructions:
#     - To use this app, please provide your OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys).
#     - Enter your question in the chat box below and interact with the ETF expert!
#     """
# )

# # Display uploaded images in two columns
# col1, col2 = st.columns(2)

# with col1:
#     st.image("WechatIMG843.jpg", caption="The ETF Book by Richard A. Ferri", use_column_width=True)

# with col2:
#     st.image("WechatIMG844.jpg", caption="ETF Handbook by K&L Gates", use_column_width=True)

# # Set OpenAI API key, first check if it is in environment variable, if not, check user input
# openai_api_key = os.getenv("OPENAI_API_KEY")

# if not openai_api_key:
#     openai_api_key = st.text_input("🔑 Please enter your OpenAI API Key", type="password")

# # Display information if no key is provided
# if not openai_api_key:
#     st.info("Please provide your OpenAI API key to continue.", icon="🗝️")

# # If API key is available
# else:
#     # Initialize chat history
#     if "messages" not in st.session_state:
#         st.session_state.messages = [
#             {"role": "assistant", "content": "I am an expert in ETFs. You can ask me any question you want to know about ETFs, and I will try my best to answer them."}
#         ]

#     # Function to load the copilot model
#     @st.cache_resource
#     def load_copilot():
#         return Copilot(openai_api_key, google_search_bot)

#     # Load copilot if not already loaded
#     if "chat_copilot" not in st.session_state:
#         with st.spinner("🤖 Initializing the ETF expert..."):
#             st.session_state.chat_copilot = load_copilot()

#     # Chat input box
#     with st.container():
#         prompt = st.chat_input("💬 Ask a question")

#         if prompt:  # If user provides input, add it to message history
#             st.session_state.messages.append({"role": "user", "content": prompt})

#     # Display chat history in a cleaner UI
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])

#     # Process the latest user input
#     if st.session_state.messages[-1]["role"] != "assistant":
#         with st.chat_message("assistant"):
#             with st.spinner("💬 ETF expert is thinking..."):
#                 retrieved_info, answer = st.session_state.chat_copilot.ask(
#                     prompt, messages=st.session_state.messages[:-1]
#                 )
#                 if isinstance(answer, str):
#                     st.write(answer)
#                 else:
#                     def generate():
#                         for chunk in answer:
#                             content = chunk.choices[0].delta.content
#                             if content:
#                                 yield content
#                     answer = st.write_stream(generate())

#             st.success("✅ Response received!")
#             st.session_state.messages.append({"role": "assistant", "content": answer})

# # Additional styling options
# st.markdown(
#     """
#     <style>
#     .stChatMessage {
#         padding: 10px;
#         margin: 10px;
#         border-radius: 10px;
#         border: 1px solid #f0f0f0;
#     }
#     </style>
#     """, unsafe_allow_html=True
# )


# import streamlit as st
# from copilot import Copilot, GoogleSearchBot
# import os

# # Set the page configuration
# st.set_page_config(
#     page_title="ETF Expert Chat",
#     page_icon="💬",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # Title and instructions
# st.title("💬 Chat with an ETF Expert")
# st.markdown(
#     """
#     Welcome to the ETF expert chatbot! Ask any questions you have about Exchange-Traded Funds (ETFs) and get real-time insights.
#     ### Instructions:
#     - To use this app, please provide your OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys).
#     - Enter your Google API Key and Google Custom Search Engine ID (CSE ID) as well to enable web search features.
#     - Enter your question in the chat box below and interact with the ETF expert!
#     """
# )

# # Display uploaded images in two columns
# col1, col2 = st.columns(2)

# with col1:
#     st.image("WechatIMG843.jpg", caption="The ETF Book by Richard A. Ferri", use_column_width=True)

# with col2:
#     st.image("WechatIMG844.jpg", caption="ETF Handbook by K&L Gates", use_column_width=True)

# # Set OpenAI API key, first check if it is in environment variable, if not, check user input
# openai_api_key = os.getenv("OPENAI_API_KEY")
# google_api_key = os.getenv("GOOGLE_API_KEY")
# google_cse_id = os.getenv("GOOGLE_CSE_ID")

# if not openai_api_key:
#     openai_api_key = st.text_input("🔑 Please enter your OpenAI API Key", type="password")

# if not google_api_key:
#     google_api_key = st.text_input("🔑 Please enter your Google API Key", type="password")

# if not google_cse_id:
#     google_cse_id = st.text_input("🔍 Please enter your Google Custom Search Engine ID (CSE ID)")

# # Display information if no keys are provided
# if not openai_api_key or not google_api_key or not google_cse_id:
#     st.info("Please provide all API keys to continue.", icon="🗝️")

# # If API keys are available
# else:
#     # Initialize chat history
#     if "messages" not in st.session_state:
#         st.session_state.messages = [
#             {"role": "assistant", "content": "I am an expert in ETFs. You can ask me any question you want to know about ETFs, and I will try my best to answer them."}
#         ]

#     # Function to load the copilot model
#     @st.cache_resource
#     def load_copilot():
#         google_search_bot = GoogleSearchBot(google_api_key, google_cse_id)
#         return Copilot(openai_api_key, google_search_bot)

#     # Load copilot if not already loaded
#     if "chat_copilot" not in st.session_state:
#         with st.spinner("🤖 Initializing the ETF expert..."):
#             st.session_state.chat_copilot = load_copilot()

#     # Chat input box
#     with st.container():
#         prompt = st.chat_input("💬 Ask a question")

#         if prompt:  # If user provides input, add it to message history
#             st.session_state.messages.append({"role": "user", "content": prompt})

#     # Display chat history in a cleaner UI
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])

#     # Process the latest user input
#     if st.session_state.messages[-1]["role"] != "assistant":
#         with st.chat_message("assistant"):
#             with st.spinner("💬 ETF expert is thinking..."):
#                 retrieved_info, answer = st.session_state.chat_copilot.ask(
#                     prompt, messages=st.session_state.messages[:-1]
#                 )
#                 if isinstance(answer, str):
#                     st.write(answer)
#                 else:
#                     def generate():
#                         for chunk in answer:
#                             content = chunk.choices[0].delta.content
#                             if content:
#                                 yield content
#                     answer = st.write_stream(generate())

#             st.success("✅ Response received!")
#             st.session_state.messages.append({"role": "assistant", "content": answer})

# # Additional styling options
# st.markdown(
#     """
#     <style>
#     .stChatMessage {
#         padding: 10px;
#         margin: 10px;
#         border-radius: 10px;
#         border: 1px solid #f0f0f0;
#     }
#     </style>
#     """, unsafe_allow_html=True
# )


# import streamlit as st
# from copilot import Copilot
# import os
# ### set openai key, first check if it is in environment variable, if not, check if it is in streamlit secrets, if not, raise error


# st.title("Chat with ETF Expert")
# st.write(
#     "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). You also need to provide a Google API key, and a Google Search ID, which you can get [here] (https://developers.google.com/custom-search/v1/overview)."
# )

# # Get OpenAI, Google API keys, and Google CSE ID from environment variables
# openai_api_key = os.getenv("OPENAI_API_KEY")
# google_api_key = os.getenv("GOOGLE_API_KEY")
# google_cse_id = os.getenv("GOOGLE_CSE_ID")

# # Get OpenAI API key from user input if it's not set
# if not openai_api_key:
#     openai_api_key = st.text_input("Please enter your OpenAI API Key", type="password")

# # Get Google API key from user input if it's not set
# if not google_api_key:
#     google_api_key = st.text_input("Please enter your Google API Key", type="password")

# # Get Google CSE ID from user input if it's not set
# if not google_cse_id:
#     google_cse_id = st.text_input("Please enter your Google CSE ID", type="password")

# # Remind the user to add any missing keys
# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.", icon="🗝️")
# elif not google_api_key:
#     st.info("Please add your Google API key to continue.", icon="🗝️")
# elif not google_cse_id:
#     st.info("Please add your Google CSE ID to continue.", icon="🗝️")
# else:
#     # Initialize the chat messages history only if all three keys are in place
#     if "messages" not in st.session_state:
#         st.session_state.messages = [
#             {"role": "assistant", "content": "I am ETF Expert, your personal assistant. You can ask me about ETFs."}
#         ]
#     @st.cache_resource
#     def load_copilot():
#         return Copilot(openai_api_key, google_api_key, google_cse_id)



#     if "chat_copilot" not in st.session_state.keys():  # Initialize the chat engine
#         st.session_state.chat_copilot = load_copilot()

#     if prompt := st.chat_input(
#         "Ask a question"
#     ):  # Prompt for user input and save to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})

#     for message in st.session_state.messages:  # Write message history to UI
#         with st.chat_message(message["role"]):
#             st.write(message["content"])

#     # If last message is not from assistant, generate a new response
#     if st.session_state.messages[-1]["role"] != "assistant":
#         with st.chat_message("assistant"):

#             retrieved_info, answer = st.session_state.chat_copilot.ask(prompt, messages=st.session_state.messages[:-1])

#             #print(retrived_info)
#             if isinstance(answer, str):
#                 st.write(answer)
#             else:
#                 ### write stream answer to UI
#                 def generate():
#                     for chunk in answer:
#                         content = chunk.choices[0].delta.content
#                         if content:
#                             yield content
#                 answer = st.write_stream(generate())

#             st.session_state.messages.append({"role": "assistant", "content": answer})

# import streamlit as st
# from copilot import Copilot
# import os

# # Streamlit app setup
# st.title("Chat with ETF Expert")
# st.write(
#     "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
#     "You also need to provide a Google API key and a Google Search ID, which you can get [here](https://developers.google.com/custom-search/v1/overview)."
# )

# # Get API keys and IDs from environment variables or user input
# openai_api_key = os.getenv("OPENAI_API_KEY") or st.text_input("Please enter your OpenAI API Key", type="password")
# google_api_key = os.getenv("GOOGLE_API_KEY") or st.text_input("Please enter your Google API Key", type="password")
# google_cse_id = os.getenv("GOOGLE_CSE_ID") or st.text_input("Please enter your Google CSE ID", type="password")

# # Check if all keys are provided
# if not openai_api_key or not google_api_key or not google_cse_id:
#     st.info("Please add all API keys to continue.", icon="🗝️")
# else:
#     # Initialize chat history if not already done
#     if "messages" not in st.session_state:
#         st.session_state.messages = [
#             {"role": "assistant", "content": "I am ETF Expert, your personal assistant. You can ask me about ETFs."}
#         ]

#     @st.cache_resource
#     def load_copilot():
#         return Copilot(openai_api_key, google_api_key, google_cse_id)

#     # Initialize Copilot if not already done
#     if "chat_copilot" not in st.session_state:
#         st.session_state.chat_copilot = load_copilot()

#     # Get user input
#     if prompt := st.chat_input("Ask a question"):
#         st.session_state.messages.append({"role": "user", "content": prompt})

#     # Display message history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])

#     # Generate a new response if the last message is from the user
#     if st.session_state.messages[-1]["role"] != "assistant":
#         with st.chat_message("assistant"):
#             placeholder = st.empty()  # Placeholder for streaming output

#             # Call the Copilot's ask method to get the response
#             retrieved_info, answer = st.session_state.chat_copilot.ask(
#                 prompt, messages=st.session_state.messages[:-1]
#             )

#             # Check if the answer is a string (non-streamed) or a stream of chunks
#             if isinstance(answer, str):
#                 # Handle non-streamed response (plain string)
#                 placeholder.markdown(answer)
#                 st.session_state.messages.append({"role": "assistant", "content": answer})
#             else:
#                 # Handle streamed response
#                 def generate_content():
#                     for chunk in answer:
#                         # Ensure chunk has the expected structure
#                         if hasattr(chunk, "choices") and chunk.choices:
#                             content = chunk.choices[0].delta.content
#                             if content:
#                                 yield content

#                 # Stream content word-by-word to the placeholder
#                 streamed_response = ""  # To accumulate the streamed content

#                 for word in generate_content():  # Iterate over streamed words
#                     streamed_response += word
#                     placeholder.markdown(streamed_response)  # Update the placeholder in real-time

#                 # Save the streamed response to the session state
#                 st.session_state.messages.append({"role": "assistant", "content": streamed_response})

import streamlit as st
from copilot import Copilot
import os

# Apply custom CSS for general styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        margin-bottom: 10px;
        color: #6c757d;
    }
    img {
        height: 500px;  /* Fixed height for uniformity */
        width: auto;  /* Maintain aspect ratio */
        object-fit: cover;  /* Cover the area while preserving aspect ratio */
        border-radius: 10px;  /* Add rounded corners */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);  /* Subtle shadow for aesthetics */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and subtitle
st.markdown('<div class="title">Chat with ETF Expert</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Provide your API keys and start chatting with the expert on ETFs!</div>',
    unsafe_allow_html=True,
)

# Use st.columns to ensure side-by-side layout
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.image("WechatIMG843.jpg", use_column_width=True)

with col2:
    st.image("WechatIMG844.jpg", use_column_width=True)

# Get API keys and IDs from environment variables or user input
openai_api_key = os.getenv("OPENAI_API_KEY") or st.text_input(
    "Please enter your OpenAI API Key", type="password", key="openai_key"
)
google_api_key = os.getenv("GOOGLE_API_KEY") or st.text_input(
    "Please enter your Google API Key", type="password", key="google_key"
)
google_cse_id = os.getenv("GOOGLE_CSE_ID") or st.text_input(
    "Please enter your Google CSE ID", type="password", key="cse_id"
)

# Check if all keys are provided
if not openai_api_key or not google_api_key or not google_cse_id:
    st.info("Please add all API keys to continue.", icon="🗝️")
else:
    # Initialize chat history if not already done
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "I am ETF Expert, your personal assistant. You can ask me about ETFs."}
        ]

    @st.cache_resource
    def load_copilot():
        return Copilot(openai_api_key, google_api_key, google_cse_id)

    # Initialize Copilot if not already done
    if "chat_copilot" not in st.session_state:
        st.session_state.chat_copilot = load_copilot()

    # Get user input
    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Generate a new response if the last message is from the user
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            placeholder = st.empty()

            # Call the Copilot's ask method to get the response
            retrieved_info, answer = st.session_state.chat_copilot.ask(
                prompt, messages=st.session_state.messages[:-1]
            )

            # Stream the response if needed
            if isinstance(answer, str):
                placeholder.markdown(f"**{answer}**")
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                def generate_content():
                    for chunk in answer:
                        if hasattr(chunk, "choices") and chunk.choices:
                            content = chunk.choices[0].delta.content
                            if content:
                                yield content

                streamed_response = ""
                for word in generate_content():
                    streamed_response += word
                    placeholder.markdown(f"**{streamed_response}**")

                st.session_state.messages.append({"role": "assistant", "content": streamed_response})
