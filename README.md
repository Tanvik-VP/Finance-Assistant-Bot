# Finance-Assistant-Bot


## Introduction

Access to accurate and timely information is essential in the rapidly evolving corporate landscape. Recognizing this requirement, the Financial Assistant Chatbot was created with the intention of completely changing the way employees interact with and access internal business data at market-leading organizations such as Disney, JP Morgan, and Berkshire Hathaway. This project's primary goal was to create a sophisticated, user-friendly application that leverages state-of-the-art natural language processing (NLP) and data retrieval technologies.
Thus, our chatbot is more than simply a tool; it's a digital assistant designed to meet the unique requirements of staff members. It distinguishes itself by providing a smooth interface that staff members may utilize to query a variety of internal data, including financial reports, corporate guidelines, and employee-specific data. Workers won't need to pick up any new systems or user interfaces because the chatbot can comprehend and handle natural language inquiries. Users can communicate with the chatbot in the same way they would with a human coworker, which makes the procedure simple and effective.
All things considered, this initiative is a major advancement in the field of corporate and financial information management. It sets a new standard for digital support in the business sector and demonstrates how AI and machine learning technologies may be used to improve workplace efficiency and information accessibility.

## Features
•	Document Ingestion: Automatically loads and processes PDF documents from a directory.
•	Advanced Text Splitting: Splits documents into manageable chunks to enhance embedding quality.
•	Embedding Generation: Utilizes HuggingFace's Sentence Transformers to generate document embeddings.
•	Efficient Vector Database: Employs FAISS for an efficient similarity search within the embeddings.
•	Retrieval-based QA: Uses RetrievalQA for fetching contextually relevant answers based on similarity.
•	Chainlit Integration: Provides a user-friendly chat-based interface for interactive usage.

## Setup
Prerequisites
1.	Python 3.7 or later.
2.	Install chain lit using pip install chainlit
3.	Install faiss using pip install faiss-cpu
4.	Install langchain using pip install langchain
Note: Check requirements.txt in the code folder and install all the dependent packages to run this code, although vectors are provided as a part of the code, if you want to reproduce this project with other knowledge base, check final report Implementation section to build the embeddings and creating your own knowledge base and using open ai or any LLM models to get your chatbot working.
