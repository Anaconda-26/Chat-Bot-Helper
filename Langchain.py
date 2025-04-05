from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_docling import DoclingLoader


# Initialize the LLM
llm = ChatOllama(model="llama3.1", temperature=0)
llm2 = ChatOllama(model="llama3.2", temperature=0)
# Define system instructions
messages = [SystemMessage(content="You are a helpful assistant for students in a university. Your main task is to help them with studies and theory. "
                                  "But u should not give them direct answers, e.g.:'What is a function of a factorial?', your answer should be like this: 'I cannot give u the function, but u may consider using a cycle in your program'. "
                                  "If they ask about theory, like 'What is Python?', don't hesitate to answer as usual. ")]


messages2 = [SystemMessage(content="You are a supervisor, that does not allow any kind of code to pass through your answer."
                                   "You will get answers of your previous collegue-model, that responds to students and helps them with their studies."
                                   "If you don't find anything like code in the answer or a solution to a math problem, DO NOT change the answer. Leave it like that with no adjustments."
                                   "When you realize that the input contains code, you should detect it and convert to an advice."
                                   "For example, you get input 'The way to solve the problem is code {code}'. Ur task is to convert the code in words, e.g. u turn into 'u can use a cycle in order to solve your problem'."
                                    "Another example: u got this answer 'Ok, Python is a programming language', ur actions would be: no changing answer, so the output is 'Python is a programming language' "
                           )]

# FILE_PATH = 'C:\\Users\\User\\Downloads\\Рекурсия.pptx'
#
# loader = DoclingLoader(file_path=FILE_PATH)
# embeddings = OllamaEmbeddings(model="llama3.1")
# vector_store = Chroma(embedding_function=embeddings)

# Example documents to store
# documents = [
#     "Cats are small, domesticated carnivorous mammals.",
#     "Cats are bigger than a horse but smaller that a pig."
#     "Dogs are loyal animals and great companions.",
#     "People often prefer cats or dogs as pets.",
# ]
# query = "What do you know about size of cats?"
# # Add documents to the vector store
# vector_store.add_texts(documents)
# retrieved_docs = vector_store.similarity_search(query, k=2)
# Load documents
# loader = WebBaseLoader("https://example.com/your-document")


# documents = loader.load()
#
# # Split documents into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
# splits = text_splitter.split_documents(documents)
#
# prompt = ChatPromptTemplate.from_template("""
# Answer the question based only on the following context:
# {context}
#
# Question: {question}
# """)
# # embeddings = OllamaEmbeddings(model='llama3.1')
#
# # Create vectorstore
# vectorstore = Chroma.from_documents(
#     documents=splits,
#     embedding=embeddings
# )
#
# # Create retriever
# retriever = vectorstore.as_retriever()
# # Create RAG chain
# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )
# response = rag_chain.invoke("How many types of bpmn models are?")
# print(response)
# f = open("C:\\Users\\User\\PycharmProjects\\TG_Bot\\2.txt", "a", encoding="utf-8")
# f.write('hi')
while True:
    user_input = input("You: ")  # Get user input
    # f.write(user_input + '\n')
    if user_input.lower() in ["exit", "quit"]:  # Allow user to exit
        print("Goodbye!")
        break

    messages.append(HumanMessage(content=user_input))  # Add user message
    ai_response =llm.invoke(messages)
    messages2.append(HumanMessage(content=ai_response.content))
    ai_response2 = llm.invoke(messages2)  # Get response
    # f.write(ai_response.content + '\n')
    print(f"AI: {ai_response.content}")
    print(f"AI2: {ai_response2.content}")
    messages.append(AIMessage(content=ai_response.content))  # Keep conversation context
# f.close()
# for doc in retrieved_docs:
#     print("Retrieved:", doc.page_content)