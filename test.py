from langchain_community.llms import VLLM
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import time
# Instantiate the VLLM
llm_q = VLLM(
    model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    trust_remote_code=True,
    max_new_tokens=512,
    vllm_kwargs={"quantization": "awq"},
)

# Define the template for the prompt
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

# Create the LLMChain with the instantiated VLLM
llm_chain = LLMChain(prompt=prompt, llm=llm_q)

# Example question

question = "Who was the US president in the year the first Pokemon game was released?"
time_start = time.time()
print(question)
reponse = llm_chain.invoke(question)
time_stop =time.time()
execution_time = time_stop - time_start

# Append the response time to the response
response_with_time = f"{reponse} Response time: {execution_time:.2f} seconds"

# Print the response along with the execution time
print(response_with_time)