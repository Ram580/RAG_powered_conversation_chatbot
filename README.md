# Upgraded RAG-powered Conversational Chatbot
### Web APP link : https://ragpoweredconversationchatbot-ago776nvsk3yvp2xg6gbapp.streamlit.app/
### Hugging face spaces app link : https://huggingface.co/spaces/ramhemanth580/Conversation_Chatbot_2.0

## Introduction:

This project showcases a cutting-edge upgraded Retrieval-Augmented Generation (RAG) application designed to empower users with intuitive, context-driven information retrieval from uploaded documents. By leveraging the high quality LLMs and  Advanced features of Langchain, this application seamlessly blends document understanding with natural language interaction.

## Key Features:

### Document Ingestion and Preprocessing:
- Effortlessly upload PDF documents. The application parses them using the PyPDF2 library, extracting the text content.
  
### Text Chunking:
- To optimize processing and storage efficiency, the extracted text is chunked into smaller segments using RecursiveCharacterTextSplitter from Langchain.
  
### Text Embedding with Sentence Transformers:
- Each text chunk is converted into a high-dimensional vector representation using Sentence Transformers' pre-trained all-MiniLM-L6-v2 model. This allows for efficient document similarity search.
  
### Pinecone Vector Database Integration:
- Pinecone, a high-performance vector database, is used to store the generated document embeddings. This enables fast retrieval of relevant documents based on user queries.
  
### Conversational Search with Memory:
- Users interact with the application through a Streamlit-built interface, asking questions in natural language.
- The application leverages a ConversationChain along with ConversationBufferWindowMemory with a window of N that power the conversational feature of the chatbot.
- The Langchain ConversationBufferWindowMemory component maintains conversation history upto past N conversations, allowing the application to consider previous interactions when responding to follow-up questions.

### efficient context window handling for responses:
- The App summarizes the previous conversation and current user input into a single query. This allows for more accurate search results.
- The context for the user question is extracted from the pinecone Index and it is sent to a conversation chain along with the refined query to get the optimal response from the LLM.
  
### Large Language Model Integration:
- The retrieved documents and conversation history are used to contextually tailor the response. The application employs the Gemini Pro LLM from Google to generate High quality informative answers to user questions.
  
### User-Friendly Interface:
- Streamlit provides a clean and intuitive interface for document upload, question input, and response display.

## Advanced Features:

### Query Refiner
- The App leverages a Query refining feature that will extract the past N conversations history and pass it to the LLM along with the current user input ,which then will summirize the conversation history and user question to create a refined Query 
- The refined query will enable the LLM to efficiently tackle the ambiguous questions from the user.
- The context for the user question is extracted from the pinecone Index and it is sent to a conversation chain along with the refined query to get the optimal response from the LLM.

### Sliding window for efficiently handling the context :
- The App efficiently utilizes the Query refiner to handle the constraints of the context window, thereby improving the accuracy of reposes.

### Converation kind of Interface:
- Users can review the conversation to stay on track and avoid repeating questions.

### Quicker Response Time:
- By utilizing high performance LLMs like Google Gemini Pro, the response time is significantly reduced compared to open source instrut LLMs which requires high compute.
  
## Technical Deep Dive:

This project demonstrates proficiency in various technical aspects:

- **Natural Language Processing (NLP):**
  - Document parsing and text extraction from PDFs using PyPDF2
  - Text Chunking with RecursiveCharacterTextSplitter
  - Text Embedding with Sentence Transformers
  
- **Vector Database Integration:**
  - Utilizing Pinecone for efficient document similarity search.
  
- **Large Language Models (LLMs):**
  - Integrating and fine-tuning a pre-trained LLM (mistralai/Mixtral-8x7B-Instruct-v0.1) for context-aware response generation.
  
- **Conversational AI:**
  - Building a Conversational Retrieval Chain with memory capabilities using Langchain libraries.
  
- **Web Development Framework:**
  - Utilizing Streamlit for rapid development of a user-friendly web application interface.

## Benefits:

- **Efficient Document Search:**
  - Quickly access information from uploaded documents through natural language queries.
  
- **Conversational Exploration:**
  - Refine your understanding by engaging in a back-and-forth dialogue with the application.
  
- **Contextual Awareness:**
  - Uncover insights from documents tailored to your specific questions.
  
- **Improved Decision-Making:**
  - Gain the knowledge you need to make informed choices based on in-depth document analysis.
