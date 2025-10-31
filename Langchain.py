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
llm2 = ChatOllama(model="llama3.1", temperature=0)
# Define system instructions
messages = [SystemMessage(content="You are a helpful assistant for students in a university. Your main task is to help them with studies and theory. "
                                  "But u should not give them direct answers, e.g.:'What is a function of a factorial?', your answer should be like this: 'I cannot give u the function, but u may consider using a cycle in your program'. "
                                  "If they ask about theory, like 'What is Python?', don't hesitate to answer as usual. ")]


messages2 = [SystemMessage(content="You are a supervisor, that does not allow any kind of code to pass through your answer."
                                   "You will get answers of your previous collegue-model, that responds to students and helps them with their studies."
                                   "If you don't find code in the answer or a solution to a math problem, DO NOT change the answer. Leave it like that with no adjustments."
                                   "When you realize that the input contains code, you should detect it and convert to an advice."
                                   "For example, you get input 'The way to solve the problem is code {code}'. Ur task is to convert the code in words, e.g. u turn into 'u can use a cycle in order to solve your problem'."
                                    "Another example: u got this answer 'Ok, Python is a programming language', ur actions would be: 'Python is a programming language' "
                           )]

messages0 = [SystemMessage(content="You are a helpful assistant for students in a university. Your main task is to help students with studies. "
                                   "Tell them everything they want to know.")]
while True:
    user_input = input("You: ")  # Get user input
    # f.write(user_input + '\n')
    if user_input.lower() in ["exit", "quit"]:  # Allow user to exit
        print("Goodbye!")
        break

    messages.append(HumanMessage(content=user_input))  # Add user message
    ai_response =llm.invoke(messages)
    # messages2.append(HumanMessage(content=ai_response.content))
    # ai_response2 = llm.invoke(messages2)  # Get response
    # f.write(ai_response.content + '\n')
    print(f"AI: {ai_response.content}")
    # print(f"AI2: {ai_response2.content}")
    messages.append(AIMessage(content=ai_response.content))  # Keep conversation context
# f.close()

