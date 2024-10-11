
# from openai import OpenAI
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from tenacity import retry, wait_random_exponential, stop_after_attempt
# import os
# @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(5))
# def chat_completion_request(client, messages, model="gpt-4o",
#                             **kwargs):
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             **kwargs
#         )
#         return response
#     except Exception as e:
#         print("Unable to generate ChatCompletion response")
#         print(f"Exception: {e}")
#         return e

# class Copilot:
#     def __init__(self, key):
#         reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
#         docs = reader.load_data()
#         embedding_model = HuggingFaceEmbedding(
#             model_name="BAAI/bge-small-en"
#         )
#         self.index = VectorStoreIndex.from_documents(docs, embed_model = embedding_model,
#                                                      show_progress=True)
#         self.retriever = self.index.as_retriever(
#                         similarity_top_k=3
#                         )

#         self.llm_client = OpenAI(api_key = key)
        
#         self.system_prompt = """
#             You are an expert on ETFs and your job is to answer questions 
#             about the ETFs.
#         """

#     def ask(self, question, messages):
#         ### use the retriever to get the answer
#         nodes = self.retriever.retrieve(question)
#         ### make answer a string with "1. <>, 2. <>, 3. <>"
#         retrieved_info = "\n".join([f"{i+1}. {node.text}" for i, node in enumerate(nodes)])
        

#         processed_query_prompt = """
#             The user is asking a question: {question}

#             The retrived information is: {retrieved_info}

#             Please answer the question based on the retrieved information. If the question is not related to ETFs, 
#             please tell the user and ask for a question related to ETFs.

#             Please highlight the information with bold text and bullet points.
#         """
        
#         processed_query = processed_query_prompt.format(question=question, 
#                                                         retrieved_info=retrieved_info)
        
#         messages = [{"role": "system", "content": self.system_prompt}] + messages + [{"role": "user", "content": processed_query}]
#         response = chat_completion_request(self.llm_client, 
#                                            messages = messages, 
#                                            stream=True)
        
#         return retrieved_info, response

# if __name__ == "__main__":
#     ### get openai key from user input
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     if not openai_api_key:
#         openai_api_key = input("Please enter your OpenAI API Key (or set it as an environment variable OPENAI_API_KEY): ")
#     copilot = Copilot(key = openai_api_key)
#     messages = []
#     while True:
#         question = input("Please ask a question: ")
#         retrived_info, answer = copilot.ask(question, messages=messages)
#         ### answer can be a generator or a string

#         #print(retrived_info)
#         if isinstance(answer, str):
#             print(answer)
#         else:
#             answer_str = ""
#             for chunk in answer:
#                 content = chunk.choices[0].delta.content
#                 if content:
#                     answer_str += content
#                     print(content, end="", flush=True)
#             print()
#             answer = answer_str

#         messages.append({"role": "user", "content": question})
#         messages.append({"role": "assistant", "content": answer})

# import requests
# import os
# import re
# from openai import OpenAI
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from tenacity import retry, wait_random_exponential, stop_after_attempt

# @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(5))
# def chat_completion_request(client, messages, model="gpt-4o", **kwargs):
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             **kwargs
#         )
#         return response
#     except Exception as e:
#         print("Unable to generate ChatCompletion response")
#         print(f"Exception: {e}")
#         return e

# # Define the GoogleSearchBot class
# class GoogleSearchBot:
#     def __init__(self, google_api_key, google_cse_id):
#         self.google_api_key = google_api_key
#         self.google_cse_id = google_cse_id

#     def search(self, query):
#         """Use Google Search API to search the web based on the user query."""
#         search_url = f"https://www.googleapis.com/customsearch/v1"
#         params = {
#             "key": self.google_api_key,
#             "cx": self.google_cse_id,
#             "q": query
#         }
#         try:
#             response = requests.get(search_url, params=params)
#             response.raise_for_status()  # This will raise an error for 4XX/5XX responses
#             search_results = response.json()
#             if "items" in search_results:
#                 results = search_results["items"]
#                 top_results = "\n".join([f"{i+1}. {item['title']}: {item['link']}" for i, item in enumerate(results[:5])])
#                 return top_results
#             else:
#                 return "No relevant results found on Google."
#         except requests.exceptions.RequestException as e:
#             # Log the error for more visibility
#             return f"Error during Google Search: {e}"

# # Define the Copilot class that integrates ETF document retrieval and Google Search
# class Copilot:
#     def __init__(self, openai_key, google_search_bot):
#         reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
#         docs = reader.load_data()
#         embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
#         self.index = VectorStoreIndex.from_documents(docs, embed_model=embedding_model, show_progress=True)
#         self.retriever = self.index.as_retriever(similarity_top_k=3)

#         self.llm_client = OpenAI(api_key=openai_key)
#         self.google_search_bot = google_search_bot
#         self.system_prompt = """
#             You are an expert on ETFs and your job is to answer questions 
#             about ETFs.
#         """

#     def ask(self, question, messages):
#         # First, try to get an answer from the ETF documents
#         nodes = self.retriever.retrieve(question)
#         retrieved_info = "\n".join([f"{i+1}. {node.text}" for i, node in enumerate(nodes)])

#         # Log the retrieved information to understand what is happening
#         print("Retrieved Info from ETF documents:", retrieved_info)

#         # Define regex patterns that signal irrelevance, allowing extra words in between
#         irrelevant_response_patterns = [
#             r"does.*not.*provide", 
#             r"does.*not.*mention", 
#             r"does.*not.*include", 
#             r"not related", 
#             r"no.*insights", 
#             r"focuses.*on", 
#             r"doesn't.*answer", 
#             r"unrelated"
#         ]
#           # Limit the length of the ETF content to avoid too much redundancy
#         MAX_LENGTH = 500  # Set a maximum length for the retrieved ETF content
#         if len(retrieved_info) > MAX_LENGTH:
#             retrieved_info = retrieved_info[:MAX_LENGTH] + "...\n[Content truncated]"

#         # Check if the retrieved info matches any of these irrelevant patterns
#         is_irrelevant = any(re.search(pattern, retrieved_info.lower()) for pattern in irrelevant_response_patterns)

#         # Debug: Log whether the response is considered irrelevant
#         print(f"Is the retrieved info irrelevant? {is_irrelevant}")

#         # Trigger Google Search if the retrieved information is irrelevant
#         if not retrieved_info.strip() or is_irrelevant:
#             print("No relevant info in ETF documents, switching to Google Search.")
#             google_results = self.google_search_bot.search(query=question)  # Correctly use the GoogleSearchBot instance
#             retrieved_info = f"Google Search Results:\n{google_results}"

#         processed_query_prompt = f"""
#             The user is asking a question: {question}

#             The retrieved information is: {retrieved_info}

#             Please answer the question based on the retrieved information. If the question is not related to ETFs, 
#             please tell the user and ask for a question related to ETFs.

#             Please highlight the information with bold text and bullet points.
#         """

#         # Send the query to OpenAI for processing
#         messages = [{"role": "system", "content": self.system_prompt}] + messages + [{"role": "user", "content": processed_query_prompt}]
#         response = chat_completion_request(self.llm_client, messages=messages, stream=True)

#         return retrieved_info, response


# if __name__ == "__main__":
#     # Get OpenAI and Google API credentials
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     if not openai_api_key:
#         openai_api_key = input("Please enter your OpenAI API Key: ")

#     google_api_key = os.getenv("GOOGLE_API_KEY")
#     if not google_api_key:
#         google_api_key = input("Please enter your Google API Key: ")

#     google_cse_id = os.getenv("GOOGLE_CSE_ID")
#     if not google_cse_id:
#         google_cse_id = input("Please enter your Google Custom Search Engine ID (CSE ID): ")

#     # Initialize Google Search Bot
#     google_search_bot = GoogleSearchBot(google_api_key, google_cse_id)

#     # Initialize Copilot with OpenAI and Google Search integration
#     copilot = Copilot(openai_key=openai_api_key, google_search_bot=google_search_bot)
#     messages = []

#     while True:
#         # Get user question
#         question = input("Please ask a question: ")
#         if question.lower() == "exit":
#             print("Exiting the Copilot...")
#             break

#         # Get response from Copilot
#         retrieved_info, answer = copilot.ask(question, messages)

#         # Check if the answer is a string or a generator and print it
#         if isinstance(answer, str):
#             print(answer)
#         else:
#             answer_str = ""
#             for chunk in answer:
#                 content = chunk.choices[0].delta.content
#                 if content:
#                     answer_str += content
#                     print(content, end="", flush=True)
#             print()
#             answer = answer_str

#         # Append the user's question and assistant's answer to the message history
#         messages.append({"role": "user", "content": question})
#         messages.append({"role": "assistant", "content": answer})



# import requests
# import os
# import re
# from openai import OpenAI
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from tenacity import retry, wait_random_exponential, stop_after_attempt

# @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(5))
# def chat_completion_request(client, messages, model="gpt-4o", **kwargs):
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             **kwargs
#         )
#         return response
#     except Exception as e:
#         print("Unable to generate ChatCompletion response")
#         print(f"Exception: {e}")
#         return e

# # Define the GoogleSearchBot class
# class GoogleSearchBot:
#     def __init__(self, google_api_key, google_cse_id):
#         self.google_api_key = google_api_key
#         self.google_cse_id = google_cse_id

#     def search(self, query):
#         """Use Google Search API to search the web based on the user query."""
#         search_url = f"https://www.googleapis.com/customsearch/v1"
#         params = {
#             "key": self.google_api_key,
#             "cx": self.google_cse_id,
#             "q": query
#         }
#         try:
#             response = requests.get(search_url, params=params)
#             response.raise_for_status()  # This will raise an error for 4XX/5XX responses
#             search_results = response.json()
#             if "items" in search_results:
#                 results = search_results["items"]
#                 top_results = "\n".join([f"{i+1}. {item['title']}: {item['link']}" for i, item in enumerate(results[:5])])
#                 return top_results
#             else:
#                 return "No relevant results found on Google."
#         except requests.exceptions.RequestException as e:
#             # Log the error for more visibility
#             return f"Error during Google Search: {e}"

# # Define the Copilot class that integrates ETF document retrieval and Google Search
# class Copilot:
#     def __init__(self, openai_key, google_search_bot):
#         reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
#         docs = reader.load_data()
#         embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
#         self.index = VectorStoreIndex.from_documents(docs, embed_model=embedding_model, show_progress=True)
#         self.retriever = self.index.as_retriever(similarity_top_k=3)

#         self.llm_client = OpenAI(api_key=openai_key)
#         self.google_search_bot = google_search_bot
#         self.system_prompt = """
#             You are an expert on ETFs and your job is to answer questions 
#             about ETFs.
#         """

#     def is_question_related_to_etf(self, question):
#         """Check if the question is related to ETFs."""
#         keywords = ['etf', 'exchange traded fund', 'funds', 'investment', 'stocks', 'bonds']
#         return any(keyword in question.lower() for keyword in keywords)

#     def ask(self, question, messages):
#         # Check if the question is related to ETFs
#         if self.is_question_related_to_etf(question):
#             print("Question is related to ETFs. Proceeding with ETF document retrieval...")
#             # Retrieve relevant content from ETF documents
#             nodes = self.retriever.retrieve(question)
#             retrieved_info = "\n".join([f"{i+1}. {node.text}" for i, node in enumerate(nodes)])

#             # Limit the length of the ETF content to avoid too much redundancy
#             MAX_LENGTH = 500  # Set a maximum length for the retrieved ETF content
#             if len(retrieved_info) > MAX_LENGTH:
#                 retrieved_info = retrieved_info[:MAX_LENGTH] + "...\n[Content truncated]"

#             # Ask the user if they want to proceed with Google Search if relevant info is found
#             print("Retrieved Info from ETF documents:", retrieved_info)

#             # Check if the retrieved info is relevant to the question
#             irrelevant_response_patterns = [
#                 r"does.*not.*provide", 
#                 r"does.*not.*mention", 
#                 r"does.*not.*include", 
#                 r"not related", 
#                 r"no.*insights", 
#                 r"focuses.*on", 
#                 r"doesn't.*answer", 
#                 r"unrelated"
#             ]
#             is_irrelevant = any(re.search(pattern, retrieved_info.lower()) for pattern in irrelevant_response_patterns)
#             print(f"Is the retrieved info irrelevant? {is_irrelevant}")

#             if is_irrelevant:
#                 print("No relevant info in ETF documents, switching to Google Search.")
#                 google_results = self.google_search_bot.search(query=question)
#                 retrieved_info = f"**Google Search Results:**\n{google_results}"
#             else:
#                 user_input = input("Do you want to conduct further search through Google? (yes/no): ")
#                 if user_input.lower() == "yes":
#                     google_results = self.google_search_bot.search(query=question)
#                     retrieved_info = f"**Google Search Results:**\n{google_results}"

#         else:
#             print("Question is not related to ETFs. Launching Google Search immediately.")
#             google_results = self.google_search_bot.search(query=question)
#             retrieved_info = f"**Google Search Results:**\n{google_results}"

#         processed_query_prompt = f"""
#             The user is asking a question: {question}

#             The retrieved information is: {retrieved_info}

#             Please answer the question based on the retrieved information. If the question is not related to ETFs, 
#             please tell the user and ask for a question related to ETFs.

#             Please highlight the information with bold text and bullet points.
#         """

#         # Send the query to OpenAI for processing
#         messages = [{"role": "system", "content": self.system_prompt}] + messages + [{"role": "user", "content": processed_query_prompt}]
#         response = chat_completion_request(self.llm_client, messages=messages, stream=True)

#         return retrieved_info, response


# if __name__ == "__main__":
#     # Get OpenAI and Google API credentials
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     if not openai_api_key:
#         openai_api_key = input("Please enter your OpenAI API Key: ")

#     google_api_key = os.getenv("GOOGLE_API_KEY")
#     if not google_api_key:
#         google_api_key = input("Please enter your Google API Key: ")

#     google_cse_id = os.getenv("GOOGLE_CSE_ID")
#     if not google_cse_id:
#         google_cse_id = input("Please enter your Google Custom Search Engine ID (CSE ID): ")

#     # Initialize Google Search Bot
#     google_search_bot = GoogleSearchBot(google_api_key, google_cse_id)

#     # Initialize Copilot with OpenAI and Google Search integration
#     copilot = Copilot(openai_key=openai_api_key, google_search_bot=google_search_bot)
#     messages = []

#     while True:
#         # Get user question
#         question = input("Please ask a question: ")
#         if question.lower() == "exit":
#             print("Exiting the Copilot...")
#             break

#         # Get response from Copilot
#         retrieved_info, answer = copilot.ask(question, messages)

#         # Check if the answer is a string or a generator and print it
#         if isinstance(answer, str):
#             print(answer)
#         else:
#             answer_str = ""
#             for chunk in answer:
#                 content = chunk.choices[0].delta.content
#                 if content:
#                     answer_str += content
#                     print(content, end="", flush=True)
#             print()
#             answer = answer_str

#         # Append the user's question and assistant's answer to the message history
#         messages.append({"role": "user", "content": question})
#         messages.append({"role": "assistant", "content": answer})




# import requests
# import os
# import re
# from openai import OpenAI
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from tenacity import retry, wait_random_exponential, stop_after_attempt

# @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(5))
# def chat_completion_request(client, messages, model="gpt-4o", **kwargs):
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             **kwargs
#         )
#         return response
#     except Exception as e:
#         print("Unable to generate ChatCompletion response")
#         print(f"Exception: {e}")
#         return e

# # Define the GoogleSearchBot class
# class GoogleSearchBot:
#     def __init__(self, google_api_key, google_cse_id):
#         self.google_api_key = google_api_key
#         self.google_cse_id = google_cse_id

#     def search(self, query):
#         """Use Google Search API to search the web based on the user query."""
#         search_url = f"https://www.googleapis.com/customsearch/v1"
#         params = {
#             "key": self.google_api_key,
#             "cx": self.google_cse_id,
#             "q": query
#         }
#         try:
#             response = requests.get(search_url, params=params)
#             response.raise_for_status()  # This will raise an error for 4XX/5XX responses
#             search_results = response.json()
#             if "items" in search_results:
#                 results = search_results["items"]
#                 top_results = "\n".join([f"{i+1}. {item['title']}: {item['link']}" for i, item in enumerate(results[:5])])
#                 sources = [item['link'] for item in results[:5]]  # Keep track of the sources
#                 return top_results, sources
#             else:
#                 return "No relevant results found on Google.", []
#         except requests.exceptions.RequestException as e:
#             # Log the error for more visibility
#             return f"Error during Google Search: {e}", []

# # Define the Copilot class that integrates ETF document retrieval and Google Search
# class Copilot:
#     def __init__(self, openai_key, google_search_bot):
#         reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
#         docs = reader.load_data()
#         embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
#         self.index = VectorStoreIndex.from_documents(docs, embed_model=embedding_model, show_progress=True)
#         self.retriever = self.index.as_retriever(similarity_top_k=3)

#         self.llm_client = OpenAI(api_key=openai_key)
#         self.google_search_bot = google_search_bot
#         self.system_prompt = """
#             You are an expert on ETFs and your job is to answer questions 
#             about ETFs.
#         """

#     def is_question_related_to_etf(self, question):
#         """Check if the question is related to ETFs."""
#         keywords = ['etf', 'exchange traded fund', 'funds', 'investment', 'stocks', 'bonds']
#         return any(keyword in question.lower() for keyword in keywords)

#     def ask(self, question, messages):
#         sources = []  # List to track sources
#         retrieved_info = ""  # Initialize retrieved_info
#         # Check if the question is related to ETFs
#         if self.is_question_related_to_etf(question):
#             print("Question is related to ETFs. Proceeding with ETF document retrieval...")
#             # Retrieve relevant content from ETF documents
#             nodes = self.retriever.retrieve(question)
#             retrieved_info = "\n".join([f"{i+1}. {node.text}" for i, node in enumerate(nodes)])

#             # Limit the length of the ETF content to avoid too much redundancy
#             MAX_LENGTH = 500  # Set a maximum length for the retrieved ETF content
#             if len(retrieved_info) > MAX_LENGTH:
#                 retrieved_info = retrieved_info[:MAX_LENGTH] + "...\n[Content truncated]"
            
#             sources.append("ETF Documents")  # Keep track of the ETF source

#             # Display the retrieved info first
#             print("\nRetrieved Information:")
#             print(retrieved_info)

#             # Check if the retrieved info is relevant to the question
#             irrelevant_response_patterns = [
#                 r"does.*not.*provide", 
#                 r"does.*not.*mention", 
#                 r"does.*not.*include", 
#                 r"not related", 
#                 r"no.*insights", 
#                 r"focuses.*on", 
#                 r"doesn't.*answer", 
#                 r"unrelated"
#             ]
#             is_irrelevant = any(re.search(pattern, retrieved_info.lower()) for pattern in irrelevant_response_patterns)
#             print(f"Is the retrieved info irrelevant? {is_irrelevant}")

#             # Now, ask the user if they want further Google search
#             if is_irrelevant:
#                 google_results, google_sources = self.google_search_bot.search(query=question)
#                 retrieved_info = f"**Google Search Results:**\n{google_results}"
#                 sources.extend(google_sources)
#             else:
#                 user_input = input("Do you want to conduct further search through Google? (yes/no): ")
#                 if user_input.lower() == "yes":
#                     google_results, google_sources = self.google_search_bot.search(query=question)
#                     retrieved_info = f"**Google Search Results:**\n{google_results}"
#                     sources.extend(google_sources)

#         else:
#             print("Question is not related to ETFs. Launching Google Search immediately.")
#             google_results, google_sources = self.google_search_bot.search(query=question)
#             retrieved_info = f"**Google Search Results:**\n{google_results}"  # Initialize retrieved_info properly here
#             print("\nGoogle Search Results:")
#             print(google_results)  # Display Google search results
#             sources.extend(google_sources)

#         # Prepare the final message to be sent to OpenAI along with sources
#         sources_text = "\n".join([f"- {source}" for source in sources])
#         processed_query_prompt = f"""
#             The user is asking a question: {question}

#             The retrieved information is: {retrieved_info}

#             Sources:
#             {sources_text}

#             Please answer the question based on the retrieved information. If the question is not related to ETFs, 
#             please tell the user and ask for a question related to ETFs.

#             Please highlight the information with bold text and bullet points.
#         """

#         # Send the query to OpenAI for processing
#         messages = [{"role": "system", "content": self.system_prompt}] + messages + [{"role": "user", "content": processed_query_prompt}]
#         response = chat_completion_request(self.llm_client, messages=messages, stream=True)

#         return retrieved_info, response


# if __name__ == "__main__":
#     # Get OpenAI and Google API credentials
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     if not openai_api_key:
#         openai_api_key = input("Please enter your OpenAI API Key: ")

#     google_api_key = os.getenv("GOOGLE_API_KEY")
#     if not google_api_key:
#         google_api_key = input("Please enter your Google API Key: ")

#     google_cse_id = os.getenv("GOOGLE_CSE_ID")
#     if not google_cse_id:
#         google_cse_id = input("Please enter your Google Custom Search Engine ID (CSE ID): ")

#     # Initialize Google Search Bot
#     google_search_bot = GoogleSearchBot(google_api_key, google_cse_id)

#     # Initialize Copilot with OpenAI and Google Search integration
#     copilot = Copilot(openai_key=openai_api_key, google_search_bot=google_search_bot)
#     messages = []

#     while True:
#         # Get user question
#         question = input("Please ask a question: ")
#         if question.lower() == "exit":
#             print("Exiting the Copilot...")
#             break

#         # Get response from Copilot
#         retrieved_info, answer = copilot.ask(question, messages)

#         # Check if the answer is a string or a generator and print it
#         if isinstance(answer, str):
#             print(answer)
#         else:
#             answer_str = ""
#             for chunk in answer:
#                 content = chunk.choices[0].delta.content
#                 if content:
#                     answer_str += content
#                     print(content, end="", flush=True)
#             print()
#             answer = answer_str

#         # Append the user's question and assistant's answer to the message history
#         messages.append({"role": "user", "content": question})
#         messages.append({"role": "assistant", "content": answer})





# import requests
# import os
# import re
# from openai import OpenAI
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from tenacity import retry, wait_random_exponential, stop_after_attempt

# @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(5))
# def chat_completion_request(client, messages, model="gpt-4o", **kwargs):
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             **kwargs
#         )
#         return response
#     except Exception as e:
#         print("Unable to generate ChatCompletion response")
#         print(f"Exception: {e}")
#         return e

# # Define the GoogleSearchBot class
# class GoogleSearchBot:
#     def __init__(self, google_api_key, google_cse_id):
#         self.google_api_key = google_api_key
#         self.google_cse_id = google_cse_id

#     def search(self, query):
#         """Use Google Search API to search the web based on the user query."""
#         search_url = f"https://www.googleapis.com/customsearch/v1"
#         params = {
#             "key": self.google_api_key,
#             "cx": self.google_cse_id,
#             "q": query
#         }
#         try:
#             response = requests.get(search_url, params=params)
#             response.raise_for_status()  # This will raise an error for 4XX/5XX responses
#             search_results = response.json()
#             if "items" in search_results:
#                 results = search_results["items"]
#                 top_results = "\n".join([f"{i+1}. {item['title']}: {item['link']}" for i, item in enumerate(results[:5])])
#                 sources = [item['link'] for item in results[:5]]  # Keep track of the sources
#                 return top_results, sources
#             else:
#                 return "No relevant results found on Google.", []
#         except requests.exceptions.RequestException as e:
#             # Log the error for more visibility
#             return f"Error during Google Search: {e}", []

# # Define the Copilot class that integrates ETF document retrieval and Google Search
# class Copilot:
#     def __init__(self, openai_key, google_search_bot):
#         reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
#         docs = reader.load_data()
#         embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
#         self.index = VectorStoreIndex.from_documents(docs, embed_model=embedding_model, show_progress=True)
#         self.retriever = self.index.as_retriever(similarity_top_k=3)

#         self.llm_client = OpenAI(api_key=openai_key)
#         self.google_search_bot = google_search_bot
#         self.system_prompt = """
#             You are an expert on ETFs and your job is to answer questions 
#             about ETFs.
#         """

#     def is_question_related_to_etf(self, question):
#         """Check if the question is related to ETFs."""
#         keywords = ['etf', 'exchange traded fund', 'funds', 'investment', 'stocks', 'bonds']
#         return any(keyword in question.lower() for keyword in keywords)

#     def anti_jailbreak_check(self, question):
#         """Basic anti-jailbreaking mechanism to prevent malicious or manipulative queries."""
#         banned_keywords = ['illegal', 'hack', 'bypass', 'jailbreak', 'exploit']
#         if any(keyword in question.lower() for keyword in banned_keywords):
#             return True  # Jailbreaking attempt detected
#         return False

#     def ask(self, question, messages):
#         sources = []  # List to track sources
#         retrieved_info = ""  # Initialize retrieved_info

#         # Perform basic anti-jailbreaking check
#         if self.anti_jailbreak_check(question):
#             return "This query is not allowed due to policy restrictions.", None

#         # Check if the question is related to ETFs
#         if self.is_question_related_to_etf(question):
#             print("Question is related to ETFs. Proceeding with ETF document retrieval...")
            
#             # Retrieve relevant content from ETF documents
#             nodes = self.retriever.retrieve(question)
            
#             # Combine retrieved content without book title and page number
#             retrieved_info = "\n\n".join([f"{i+1}. {node.text}" for i, node in enumerate(nodes)])

#             # Limit the length of the ETF content to avoid too much redundancy
#             MAX_LENGTH = 500  # Set a maximum length for the retrieved ETF content
#             if len(retrieved_info) > MAX_LENGTH:
#                 retrieved_info = retrieved_info[:MAX_LENGTH] + "...\n[Content truncated]"

#             # Track the sources from the ETF documents (content only, no book title/page number)
#             sources.extend([node.text for node in nodes])

#             # Conduct relevance check (is_relevant instead of is_irrelevant)
#             irrelevant_response_patterns = [
#                 r"does.*not.*provide", 
#                 r"does.*not.*mention", 
#                 r"does.*not.*include", 
#                 r"not related", 
#                 r"no.*insights", 
#                 r"focuses.*on", 
#                 r"doesn't.*answer", 
#                 r"unrelated"
#             ]
#             is_relevant = not any(re.search(pattern, retrieved_info.lower()) for pattern in irrelevant_response_patterns)

#             # If the content is relevant, ask the user if they want to proceed with Google search
#             if is_relevant:
#                 print("Retrieval from the ETF books completed. Do you want to conduct further search through Google? (yes/no)")
#                 user_input = input()  # Get the user's response

#                 if user_input.lower() == "yes":
#                     # Perform Google search if requested
#                     google_results, google_sources = self.google_search_bot.search(query=question)
                    
#                     # Print both ETF retrieved content and Google search results
#                     print(f"(1) Retrieved from ETF Documents:\n{retrieved_info}\n")
#                     print(f"(2) Google Search Results:\n{google_results}")

#                     sources.extend(google_sources)  # Track Google sources

#                 else:
#                     # If the user does not want Google search, print only ETF results
#                     print(f"(1) Retrieved from ETF Documents:\n{retrieved_info}\n")

#             else:
#                 # If the content is not relevant, perform Google search immediately
#                 print("ETF content is not relevant. Launching Google Search...")
#                 google_results, google_sources = self.google_search_bot.search(query=question)
#                 retrieved_info = f"**Google Search Results:**\n{google_results}"
#                 print("\nGoogle Search Results:")
#                 print(google_results)
#                 sources.extend(google_sources)

#         else:
#             print("Question is not related to ETFs. Launching Google Search immediately.")
#             google_results, google_sources = self.google_search_bot.search(query=question)
#             retrieved_info = f"**Google Search Results:**\n{google_results}"
#             print("\nGoogle Search Results:")
#             print(google_results)
#             sources.extend(google_sources)

#         # Prepare the final message to be sent to OpenAI along with sources
#         sources_text = "\n".join([f"- {source}" for source in sources])
#         processed_query_prompt = f"""
#             The user is asking a question: {question}

#             The retrieved information is: {retrieved_info}

#             Sources:
#             {sources_text}

#             Please answer the question based on the retrieved information. If the question is not related to ETFs, 
#             please tell the user and ask for a question related to ETFs.

#             Please highlight the information with bold text and bullet points.
#         """

#         # Send the query to OpenAI for processing
#         messages = [{"role": "system", "content": self.system_prompt}] + messages + [{"role": "user", "content": processed_query_prompt}]
#         response = chat_completion_request(self.llm_client, messages=messages, stream=True)

#         return retrieved_info, response




# if __name__ == "__main__":
#     # Get OpenAI and Google API credentials
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     if not openai_api_key:
#         openai_api_key = input("Please enter your OpenAI API Key: ")

#     google_api_key = os.getenv("GOOGLE_API_KEY")
#     if not google_api_key:
#         google_api_key = input("Please enter your Google API Key: ")

#     google_cse_id = os.getenv("GOOGLE_CSE_ID")
#     if not google_cse_id:
#         google_cse_id = input("Please enter your Google Custom Search Engine ID (CSE ID): ")

#     # Initialize Google Search Bot
#     google_search_bot = GoogleSearchBot(google_api_key, google_cse_id)

#     # Initialize Copilot with OpenAI and Google Search integration
#     copilot = Copilot(openai_key=openai_api_key, google_search_bot=google_search_bot)
#     messages = []

#     while True:
#         # Get user question
#         question = input("Please ask a question: ")
#         if question.lower() == "exit":
#             print("Exiting the Copilot...")
#             break

#         # Get response from Copilot
#         retrieved_info, answer = copilot.ask(question, messages)

#         # Check if the answer is a string or a generator and print it
#         if isinstance(answer, str):
#             print(answer)
#         else:
#             answer_str = ""
#             for chunk in answer:
#                 content = chunk.choices[0].delta.content
#                 if content:
#                     answer_str += content
#                     print(content, end="", flush=True)
#             print()
#             answer = answer_str

#         # Append the user's question and assistant's answer to the message history
#         messages.append({"role": "user", "content": question})
#         messages.append({"role": "assistant", "content": answer})



# import requests
# import os
# import re
# from openai import OpenAI
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from tenacity import retry, wait_random_exponential, stop_after_attempt
# from collections import defaultdict
# import time

# @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(5))
# def chat_completion_request(client, messages, model="gpt-4o", **kwargs):
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             **kwargs
#         )
#         return response
#     except Exception as e:
#         print("Unable to generate ChatCompletion response")
#         print(f"Exception: {e}")
#         return e

# # Define the GoogleSearchBot class
# class GoogleSearchBot:
#     def __init__(self, google_api_key, google_cse_id):
#         self.google_api_key = google_api_key
#         self.google_cse_id = google_cse_id

#     def search(self, query):
#         """Use Google Search API to search the web based on the user query."""
#         search_url = f"https://www.googleapis.com/customsearch/v1"
#         params = {
#             "key": self.google_api_key,
#             "cx": self.google_cse_id,
#             "q": query
#         }
#         try:
#             response = requests.get(search_url, params=params)
#             response.raise_for_status()  # This will raise an error for 4XX/5XX responses
#             search_results = response.json()
#             if "items" in search_results:
#                 results = search_results["items"]
#                 top_results = "\n".join([f"{i+1}. {item['title']}: {item['link']}" for i, item in enumerate(results[:5])])
#                 sources = [item['link'] for item in results[:5]]  # Keep track of the sources
#                 return top_results, sources
#             else:
#                 return "No relevant results found on Google.", []
#         except requests.exceptions.RequestException as e:
#             # Log the error for more visibility
#             return f"Error during Google Search: {e}", []

# # Define the Copilot class that integrates ETF document retrieval and Google Search
# class Copilot:
#     def __init__(self, openai_key, google_search_bot):
#         reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
#         docs = reader.load_data()
#         embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
#         self.index = VectorStoreIndex.from_documents(docs, embed_model=embedding_model, show_progress=True)
#         self.retriever = self.index.as_retriever(similarity_top_k=3)

#         self.llm_client = OpenAI(api_key=openai_key)
#         self.google_search_bot = google_search_bot
#         self.system_prompt = """
#             You are an expert on ETFs and your job is to answer questions 
#             about ETFs.
#         """
        
#         # Anti-jailbreaking related initialization
#         self.failed_attempts = defaultdict(int)
#         self.time_locks = defaultdict(int)
#         self.MAX_ATTEMPTS = 3
#         self.LOCK_TIME = 600  # Lock the user out for 10 minutes after 3 failed attempts

#     def is_question_related_to_etf(self, question):
#         """Check if the question is related to ETFs."""
#         keywords = ['etf', 'exchange traded fund', 'funds', 'investment', 'stocks', 'bonds']
#         return any(keyword in question.lower() for keyword in keywords)

#     def rate_limit_check(self, user_id):
#         """Check if the user has exceeded the allowed number of failed attempts."""
#         if self.failed_attempts[user_id] >= self.MAX_ATTEMPTS:
#             if time.time() - self.time_locks[user_id] < self.LOCK_TIME:
#                 return False  # User is still in the lockout period
#             else:
#                 # Reset the attempts and time lock after lockout period is over
#                 self.failed_attempts[user_id] = 0
#                 self.time_locks[user_id] = 0
#         return True

#     def anti_jailbreak_check(self, question, user_id):
#         """Enhanced anti-jailbreaking mechanism with rate limiting."""
#         # Comprehensive banned keywords list and patterns
#         banned_keywords = [
#             'illegal', 'hack', 'bypass', 'jailbreak', 'exploit', 
#             'malware', 'virus', 'phishing', 'keylogger', 'ddos',
#             'sql injection', 'buffer overflow', 'cross-site scripting'
#         ]
        
#         # Detecting patterns that could indicate malicious attempts
#         banned_patterns = [
#             r'(how to|ways to|steps to|methods to).* (hack|exploit|bypass|jailbreak)',  # Complex patterns
#             r'(\bshutdown\b|\bdisable\b|\bmodify\b|\boverride\b) security',
#             r'(destroy|corrupt|delete) (files|data|logs)',
#         ]
        
#         # Rate limit check
#         if not self.rate_limit_check(user_id):
#             return "Too many failed attempts. Please wait before trying again.", None
        
#         # Check for banned keywords
#         if any(keyword in question.lower() for keyword in banned_keywords):
#             self.failed_attempts[user_id] += 1
#             if self.failed_attempts[user_id] >= self.MAX_ATTEMPTS:
#                 self.time_locks[user_id] = time.time()
#             return True  # Jailbreaking attempt detected
        
#         # Check for banned patterns using regex
#         if any(re.search(pattern, question.lower()) for pattern in banned_patterns):
#             self.failed_attempts[user_id] += 1
#             if self.failed_attempts[user_id] >= self.MAX_ATTEMPTS:
#                 self.time_locks[user_id] = time.time()
#             return True  # Jailbreaking pattern detected
        
#         return False

#     def ask(self, question, messages, user_id):
#         sources = []  # List to track sources
#         retrieved_info = ""  # Initialize retrieved_info

#         # Perform basic anti-jailbreaking check
#         if self.anti_jailbreak_check(question, user_id):
#             return "This query is not allowed due to policy restrictions.", None

#         # Check if the question is related to ETFs
#         if self.is_question_related_to_etf(question):
#             print("Question is related to ETFs. Proceeding with ETF document retrieval...")
            
#             # Retrieve relevant content from ETF documents
#             nodes = self.retriever.retrieve(question)
            
#             # Combine retrieved content without book title and page number
#             retrieved_info = "\n\n".join([f"{i+1}. {node.text}" for i, node in enumerate(nodes)])

#             # Limit the length of the ETF content to avoid too much redundancy
#             MAX_LENGTH = 1000  # Set a maximum length for the retrieved ETF content
#             if len(retrieved_info) > MAX_LENGTH:
#                 retrieved_info = retrieved_info[:MAX_LENGTH] + "...\n[Content truncated]"

#             # Track the sources from the ETF documents (content only, no book title/page number)
#             sources.extend([node.text for node in nodes])

#             # Conduct relevance check (is_relevant instead of is_irrelevant)
#             irrelevant_response_patterns = [
#                 r"does.*not.*provide", 
#                 r"does.*not.*mention", 
#                 r"does.*not.*include", 
#                 r"not related", 
#                 r"no.*insights", 
#                 r"focuses.*on", 
#                 r"doesn't.*answer", 
#                 r"unrelated"
#             ]
#             is_relevant = not any(re.search(pattern, retrieved_info.lower()) for pattern in irrelevant_response_patterns)

#             # If the content is relevant, ask the user if they want to proceed with Google search
#             if is_relevant:
#                 print("Retrieval from the ETF books completed. Do you want to conduct further search through Google? (yes/no)")
#                 user_input = input()  # Get the user's response

#                 if user_input.lower() == "yes":
#                     # Perform Google search if requested
#                     google_results, google_sources = self.google_search_bot.search(query=question)
                    
#                     # Print both ETF retrieved content and Google search results
#                     print(f"(1) Retrieved from ETF Documents:\n{retrieved_info}\n")
#                     print(f"(2) Google Search Results:\n{google_results}")

#                     sources.extend(google_sources)  # Track Google sources

#                 else:
#                     # If the user does not want Google search, print only ETF results
#                     print(f"(1) Retrieved from ETF Documents:\n{retrieved_info}\n")

#             else:
#                 # If the content is not relevant, perform Google search immediately
#                 print("ETF content is not relevant. Launching Google Search...")
#                 google_results, google_sources = self.google_search_bot.search(query=question)
#                 retrieved_info = f"**Google Search Results:**\n{google_results}"
#                 print("\nGoogle Search Results:")
#                 print(google_results)
#                 sources.extend(google_sources)

#         else:
#             print("Question is not related to ETFs. Launching Google Search immediately.")
#             google_results, google_sources = self.google_search_bot.search(query=question)
#             retrieved_info = f"**Google Search Results:**\n{google_results}"
#             print("\nGoogle Search Results:")
#             print(google_results)
#             sources.extend(google_sources)

#         # Prepare the final message to be sent to OpenAI along with sources
#         sources_text = "\n".join([f"- {source}" for source in sources])
#         processed_query_prompt = f"""
#             The user is asking a question: {question}

#             The retrieved information is: {retrieved_info}

#             Sources:
#             {sources_text}

#             Please answer the question based on the retrieved information. If the question is not related to ETFs, 
#             please tell the user and ask for a question related to ETFs.

#             Please highlight the information with bold text and bullet points.
#         """

#         # Send the query to OpenAI for processing
#         messages = [{"role": "system", "content": self.system_prompt}] + messages + [{"role": "user", "content": processed_query_prompt}]
#         response = chat_completion_request(self.llm_client, messages=messages, stream=True)

#         return retrieved_info, response

# if __name__ == "__main__":
#     # Get OpenAI and Google API credentials
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     if not openai_api_key:
#         openai_api_key = input("Please enter your OpenAI API Key: ")

#     google_api_key = os.getenv("GOOGLE_API_KEY")
#     if not google_api_key:
#         google_api_key = input("Please enter your Google API Key: ")

#     google_cse_id = os.getenv("GOOGLE_CSE_ID")
#     if not google_cse_id:
#         google_cse_id = input("Please enter your Google Custom Search Engine ID (CSE ID): ")

#     # Initialize Google Search Bot
#     google_search_bot = GoogleSearchBot(google_api_key, google_cse_id)

#     # Initialize Copilot with OpenAI and Google Search integration
#     copilot = Copilot(openai_key=openai_api_key, google_search_bot=google_search_bot)
#     messages = []

#     while True:
#         # Get user question
#         question = input("Please ask a question: ")
#         if question.lower() == "exit":
#             print("Exiting the Copilot...")
#             break

#         # Get response from Copilot
#         user_id = "user_id_1"  # Simulating a user ID for tracking purposes
#         retrieved_info, answer = copilot.ask(question, messages, user_id)

#         # Check if the answer is a string or a generator and print it
#         if isinstance(answer, str):
#             print(answer)
#         else:
#             answer_str = ""
#             for chunk in answer:
#                 content = chunk.choices[0].delta.content
#                 if content:
#                     answer_str += content
#                     print(content, end="", flush=True)
#             print()
#             answer = answer_str

#         # Append the user's question and assistant's answer to the message history
#         messages.append({"role": "user", "content": question})
#         messages.append({"role": "assistant", "content": answer})

# import requests
# import os
# import re
# from openai import OpenAI
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from tenacity import retry, wait_random_exponential, stop_after_attempt

# # Retry mechanism for OpenAI API requests
# @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(5))
# def chat_completion_request(client, messages, model="gpt-4o", **kwargs):
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             **kwargs
#         )
#         return response
#     except Exception as e:
#         print("Unable to generate ChatCompletion response")
#         print(f"Exception: {e}")
#         return e

# # Define the GoogleSearchBot class
# class GoogleSearchBot:
#     def __init__(self, google_api_key, google_cse_id):
#         self.google_api_key = google_api_key
#         self.google_cse_id = google_cse_id

#     def search(self, query):
#         """Use Google Search API to search the web based on the user query."""
#         search_url = f"https://www.googleapis.com/customsearch/v1"
#         params = {
#             "key": self.google_api_key,
#             "cx": self.google_cse_id,
#             "q": query
#         }
#         try:
#             response = requests.get(search_url, params=params)
#             response.raise_for_status()  # Raise an error for 4XX/5XX responses
#             search_results = response.json()
#             if "items" in search_results:
#                 results = search_results["items"]
#                 top_results = "\n".join([f"{i+1}. {item['title']}: {item['link']}" for i, item in enumerate(results[:5])])
#                 sources = [item['link'] for item in results[:5]]  # Track sources
#                 return top_results, sources
#             else:
#                 return "No relevant results found on Google.", []
#         except requests.exceptions.RequestException as e:
#             return f"Error during Google Search: {e}", []

# # Define the Copilot class that integrates ETF document retrieval and Google Search
# class Copilot:
#     def __init__(self, openai_key, google_search_bot):
#         reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
#         docs = reader.load_data()
#         embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
#         self.index = VectorStoreIndex.from_documents(docs, embed_model=embedding_model, show_progress=True)
#         self.retriever = self.index.as_retriever(similarity_top_k=3)

#         self.llm_client = OpenAI(api_key=openai_key)
#         self.google_search_bot = google_search_bot
#         self.system_prompt = "You are an expert on ETFs. Your job is to answer questions about ETFs."

#     def is_question_related_to_etf(self, question):
#         """Check if the question is related to ETFs."""
#         keywords = ['etf', 'exchange traded fund', 'funds', 'investment', 'stocks', 'bonds']
#         return any(keyword in question.lower() for keyword in keywords)

#     def anti_jailbreak_check(self, question):
#         """Enhanced anti-jailbreaking mechanism with more comprehensive keywords and patterns."""
#         banned_keywords = [
#             'illegal', 'hack', 'bypass', 'jailbreak', 'exploit', 'malware', 'virus', 'phishing', 
#             'keylogger', 'ddos', 'sql injection', 'xss', 'cross-site scripting', 'remote code execution', 
#             'rce', 'buffer overflow', 'privilege escalation', 'root access', 'backdoor', 'trojan', 'worm',
#             'dark web', 'ransomware', 'data leak', 'brute force', 'cyber attack', 'botnet'
#         ]
        
#         # Detecting more complex patterns that could indicate malicious attempts
#         banned_patterns = [
#             r'(how to|ways to|steps to|methods to).* (hack|exploit|bypass|jailbreak)',  # Complex patterns
#             r'(\bshutdown\b|\bdisable\b|\bmodify\b|\boverride\b) security',
#             r'(destroy|corrupt|delete) (files|data|logs)',
#             r'(\binstall\b|\bdeploy\b).*malware',
#             r'(how to|ways to).*(\bcompromise\b|\bhijack\b) user accounts'
#         ]
        
#         # Check for banned keywords
#         if any(keyword in question.lower() for keyword in banned_keywords):
#             return True  # Jailbreaking attempt detected

#         # Check for banned patterns using regex
#         if any(re.search(pattern, question.lower()) for pattern in banned_patterns):
#             return True  # Jailbreaking pattern detected

#         return False

#     def ask(self, question, messages):
#         sources = []  # Track sources
#         retrieved_info = ""  # Initialize retrieved information

#         # Perform basic anti-jailbreaking check
#         if self.anti_jailbreak_check(question):
#             return "This query is not allowed due to policy restrictions.", None

#         # Check if the question is related to ETFs
#         if self.is_question_related_to_etf(question):
#             print("Question is related to ETFs. Proceeding with ETF document retrieval...")
            
#             # Retrieve relevant content from ETF documents
#             nodes = self.retriever.retrieve(question)
            
#             # Combine retrieved content without book title and page number
#             retrieved_info = "\n\n".join([f"{i+1}. {node.text}" for i, node in enumerate(nodes)])

#             # Limit the length of the ETF content to avoid too much redundancy
#             MAX_LENGTH = 1000  # Set a maximum length for the retrieved ETF content
#             if len(retrieved_info) > MAX_LENGTH:
#                 retrieved_info = retrieved_info[:MAX_LENGTH] + "...\n[Content truncated]"

#             # Track the sources from the ETF documents
#             sources.extend([node.text for node in nodes])

#             # Check if the retrieved content is relevant
#             irrelevant_response_patterns = [
#                 r"does.*not.*provide", r"does.*not.*mention", r"does.*not.*include", 
#                 r"not related", r"no.*insights", r"focuses.*on", r"doesn't.*answer", r"unrelated"
#             ]
#             is_relevant = not any(re.search(pattern, retrieved_info.lower()) for pattern in irrelevant_response_patterns)

#             if is_relevant:
#                 # Perform Google search if user requests it
#                 print("Retrieval from the ETF books completed. Do you want to conduct further search through Google? (yes/no)")
#                 user_input = input()

#                 if user_input.lower() == "yes":
#                     google_results, google_sources = self.google_search_bot.search(query=question)
#                     print(f"(1) Retrieved from ETF Documents:\n{retrieved_info}\n")
#                     print(f"(2) Google Search Results:\n{google_results}")
#                     sources.extend(google_sources)
#                 else:
#                     print(f"(1) Retrieved from ETF Documents:\n{retrieved_info}\n")
#             else:
#                 # If the content is not relevant, perform Google search
#                 print("ETF content is not relevant. Launching Google Search...")
#                 google_results, google_sources = self.google_search_bot.search(query=question)
#                 retrieved_info = f"**Google Search Results:**\n{google_results}"
#                 print(google_results)
#                 sources.extend(google_sources)

#         else:
#             # Question is not related to ETFs, proceed with Google Search immediately
#             print("Question is not related to ETFs. Launching Google Search immediately.")
#             google_results, google_sources = self.google_search_bot.search(query=question)
#             retrieved_info = f"**Google Search Results:**\n{google_results}"
#             print(google_results)
#             sources.extend(google_sources)

#         # Prepare the final response
#         sources_text = "\n".join([f"- {source}" for source in sources])
#         processed_query_prompt = f"""
#             The user is asking a question: {question}
#             The retrieved information is: {retrieved_info}
#             Sources:
#             {sources_text}
#             Please answer the question based on the retrieved information. If the question is not related to ETFs, 
#             please tell the user and ask for a question related to ETFs.
#         """

#         # Send the query to OpenAI for processing
#         messages = [{"role": "system", "content": self.system_prompt}] + messages + [{"role": "user", "content": processed_query_prompt}]
#         response = chat_completion_request(self.llm_client, messages=messages, stream=True)

#         return retrieved_info, response

# if __name__ == "__main__":
#     # Get OpenAI and Google API credentials
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     if not openai_api_key:
#         openai_api_key = input("Please enter your OpenAI API Key: ")

#     google_api_key = os.getenv("GOOGLE_API_KEY")
#     if not google_api_key:
#         google_api_key = input("Please enter your Google API Key: ")

#     google_cse_id = os.getenv("GOOGLE_CSE_ID")
#     if not google_cse_id:
#         google_cse_id = input("Please enter your Google Custom Search Engine ID (CSE ID): ")

#     # Initialize Google Search Bot
#     google_search_bot = GoogleSearchBot(google_api_key, google_cse_id)

#     # Initialize Copilot with OpenAI and Google Search integration
#     copilot = Copilot(openai_key=openai_api_key, google_search_bot=google_search_bot)
#     messages = []

#     while True:
#         # Get user question
#         question = input("Please ask a question: ")
#         if question.lower() == "exit":
#             print("Exiting the Copilot...")
#             break

#         # Get response from Copilot
#         retrieved_info, answer = copilot.ask(question, messages)

#         # Print the response
#         if isinstance(answer, str):
#             print(answer)
#         else:
#             answer_str = ""
#             for chunk in answer:
#                 content = chunk.choices[0].delta.content
#                 if content:
#                     answer_str += content
#                     print(content, end="", flush=True)
#             print()
#             answer = answer_str

#         # Append the user's question and assistant's answer to the message history
#         messages.append({"role": "user", "content": question})
#         messages.append({"role": "assistant", "content": answer})


import requests
import os
import re

from openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Retry mechanism for OpenAI API requests
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(5))
def chat_completion_request(client, messages, model="gpt-4o", **kwargs):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

# Define the GoogleSearchBot class
class GoogleSearchBot:
    def __init__(self, google_api_key, google_cse_id):
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id

    def search(self, query):
        """Use Google Search API to search the web based on the user query."""
        search_url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query
        }
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()  # Raise an error for 4XX/5XX responses
            search_results = response.json()
            if "items" in search_results:
                results = search_results["items"]
                top_results = "\n".join([f"{i+1}. {item['title']}: {item['link']}" for i, item in enumerate(results[:5])])
                sources = [item['link'] for item in results[:5]]  # Track sources
                return top_results, sources
            else:
                return "No relevant results found on Google.", []
        except requests.exceptions.RequestException as e:
            return f"Error during Google Search: {e}", []

# Define the Copilot class that integrates ETF document retrieval and Google Search
class Copilot:
    def __init__(self, openai_key, google_search_bot):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
        self.index = VectorStoreIndex.from_documents(docs, embed_model=embedding_model, show_progress=True)
        self.retriever = self.index.as_retriever(similarity_top_k=3)

        self.llm_client = OpenAI(api_key=openai_key)
        self.google_search_bot = google_search_bot
        self.system_prompt = "You are an expert on ETFs. Your job is to answer questions about ETFs."

    def is_question_related_to_etf(self, question):
        """Check if the question is related to ETFs."""
        keywords = ['etf', 'exchange traded fund', 'funds', 'investment', 'stocks', 'bonds']
        return any(keyword in question.lower() for keyword in keywords)

    def anti_jailbreak_check(self, question):
        """Anti-jailbreaking mechanism with comprehensive keywords and patterns."""
        banned_keywords = [
            'illegal', 'hack', 'bypass', 'jailbreak', 'exploit', 'malware', 'virus', 'phishing', 
            'keylogger', 'ddos', 'sql injection', 'cross-site scripting', 'rce', 'buffer overflow', 
            'privilege escalation', 'root access', 'backdoor', 'trojan', 'worm', 'dark web', 'ransomware'
        ]
        
        # Complex patterns that could indicate malicious attempts
        banned_patterns = [
            r'(how to|ways to|steps to|methods to).* (hack|exploit|bypass|jailbreak)',  
            r'(\bshutdown\b|\bdisable\b|\bmodify\b|\boverride\b) security',
            r'(destroy|corrupt|delete) (files|data|logs)'
        ]
        
        # Check for banned keywords
        if any(keyword in question.lower() for keyword in banned_keywords):
            return True  # Jailbreaking attempt detected

        # Check for banned patterns using regex
        if any(re.search(pattern, question.lower()) for pattern in banned_patterns):
            return True  # Jailbreaking pattern detected

        return False

    def ask(self, question, messages):
        sources = []  # Track sources
        retrieved_info = ""  # Initialize retrieved information

        # Perform basic anti-jailbreaking check
        if self.anti_jailbreak_check(question):
            return "This query is not allowed due to policy restrictions.", None

        # Check if the question is related to ETFs
        if self.is_question_related_to_etf(question):
            print("Question is related to ETFs. Proceeding with ETF document retrieval...")
            
            # Retrieve relevant content from ETF documents
            nodes = self.retriever.retrieve(question)
            
            # Combine retrieved content
            retrieved_info = "\n\n".join([f"{i+1}. {node.text}" for i, node in enumerate(nodes)])

            # Limit the length of the ETF content
            MAX_LENGTH = 1000  
            if len(retrieved_info) > MAX_LENGTH:
                retrieved_info = retrieved_info[:MAX_LENGTH] + "...\n[Content truncated]"

            # Track the sources from the ETF documents
            sources.extend([node.text for node in nodes])

            # Check if the retrieved content is relevant
            irrelevant_response_patterns = [
                r"does.*not.*provide", 
                r"does.*not.*mention", 
                r"does.*not.*include", 
                r"no.*insights", 
                r"unrelated"
            ]
            is_relevant = not any(re.search(pattern, retrieved_info.lower()) for pattern in irrelevant_response_patterns)

            if is_relevant:
                google_results, google_sources = self.google_search_bot.search(query=question)
                sources.extend(google_sources)
                print(f"(1) Retrieved from ETF Documents:\n{retrieved_info}\n")
                print(f"(2) Google Search Results:\n{google_results}")
            else:
                google_results, google_sources = self.google_search_bot.search(query=question)
                retrieved_info = f"**Google Search Results:**\n{google_results}"
                sources.extend(google_sources)

        else:
            google_results, google_sources = self.google_search_bot.search(query=question)
            retrieved_info = f"**Google Search Results:**\n{google_results}"
            print(google_results)
            sources.extend(google_sources)

        # Prepare the final message to be sent to OpenAI
        sources_text = "\n".join([f"- {source}" for source in sources])
        processed_query_prompt = f"""
            The user is asking a question: {question}

            The retrieved information is: {retrieved_info}

            Sources:
            {sources_text}

            Please answer the question based on the retrieved information.
        """

        # Send the query to OpenAI for processing
        messages = [{"role": "system", "content": self.system_prompt}] + messages + [{"role": "user", "content": processed_query_prompt}]
        response = chat_completion_request(self.llm_client, messages=messages, stream=True)

        return retrieved_info, response
