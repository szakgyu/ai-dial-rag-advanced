from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


#TODO:
# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """
You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

#TODO:
# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""


def main():
    embeddings_client = DialEmbeddingsClient(deployment_name='text-embedding-3-small-1', api_key=API_KEY)
    embeddings_search_client = DialEmbeddingsClient(deployment_name='text-embedding-005', api_key=API_KEY)
    chat_completion_client = DialChatCompletionClient(deployment_name='gpt-4', api_key=API_KEY)
    db_config = {
        'host': 'localhost',
        'port': 5433,
        'database': 'vectordb',
        'user': 'postgres',
        'password': 'postgres'
    }
    text_processor = TextProcessor(embeddings_client, db_config)
    text_search_processor = TextProcessor(embeddings_search_client, db_config)

    conversation = Conversation()

    conversation.add_message(Message(role=Role.SYSTEM, content=SYSTEM_PROMPT))

    load_context = input("\nLoad context to VectorDB (y/n)? > ").strip()
    if load_context.lower().strip() in ['y', 'yes']:
        text_processor.process_text_file(
            file_path='./task/embeddings/microwave_manual.txt',
            chunk_size=400,
            overlap=40,
            dimensions=384,
            truncate_table=True
        )
        print("="*100)

    conversation = Conversation()
    conversation.add_message(
        Message(Role.SYSTEM, SYSTEM_PROMPT)
    )

    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting chat.")
            break

        # Retrieve context using text_processor.search()
        context_chunks = text_search_processor.search(
            search_mode=SearchMode.EUCLIDIAN_DISTANCE,
            user_request=user_input,
            top_k=5,
            score_threshold=0.01,
            dimensions=384
        )

        print(f"Retrieved context chunks: {context_chunks}")  # Debugging line to check retrieved context --- IGNORE ---

        # Create system prompt with retrieved context
        system_prompt_with_context = USER_PROMPT.format(context="\n".join(context_chunks), query=user_input)


        print(f"\nSystem prompt with context:\n{system_prompt_with_context}\n")  # Debugging line to check the system prompt

        conversation.add_message(Message(role=Role.USER, content=system_prompt_with_context))

        # Generate response using chat completion client
        response = chat_completion_client.get_completion(conversation.messages)

        print(f"Assistant: {response.content}")
        conversation.add_message(Message(role=Role.AI, content=response))

main()