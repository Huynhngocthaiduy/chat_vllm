memory_prompt_template = """<s>[INST] <<SYS>>
You are an AI chatbot having a conversation with a human. You will be given a question and a context to provide information. Answer the question based as much as possible on the context provided.
<</SYS>>
Context: 
{history}
Question: {human_input} [/INST]
"""
memory_prompt_template_mistral = """[INST] You are an AI chatbot having a conversation with a human. Answer his questions.
Context: {history}
Question: {human_input}
[/INST]
"""
pdf_chat_prompt = """<s>[INST] Answer the user question based on the given context. Also consider the chat history.[/INST]
Chat history: {history}
Context: {context}
Question: {human_input}
Answer:"""