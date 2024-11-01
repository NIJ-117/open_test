import os
import time
from dotenv import load_dotenv
from huggingface_hub import login, whoami
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain import hub
from transformers import AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from optimum.intel.openvino import OVModelForCausalLM
import re
import warnings
from transformers import AutoModel


warnings.simplefilter("ignore")
load_dotenv(verbose=True)

# model_id = 'Llama-3.2-3B'
model_id='Llama-3.2-1B'
model_name        = os.getenv('MODEL_NAME')
model_precision   = os.getenv('MODEL_PRECISION', "FP16")
inference_device  = os.getenv('INFERENCE_DEVICE')
ov_config         = {"PERFORMANCE_HINT":"LATENCY", "NUM_STREAMS":"1", "CACHE_DIR":os.getenv('CACHE_DIR')}
num_max_tokens    = int(os.getenv('NUM_MAX_TOKENS', 4096))
rag_chain_type    = os.getenv('RAG_CHAIN_TYPE')
embeddings_model  = os.getenv('MODEL_EMBEDDINGS')
chroma_path       = os.getenv('CHROMA_PATH')


embeddings = HuggingFaceEmbeddings(
    model_name = embeddings_model,
    model_kwargs = {'device':'cpu'},
    encode_kwargs = {'normalize_embeddings': False}
)

prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

custom_prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use markdown to respond.
Context: {context} \n
Question: {question} \n
Answer: 
"""

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=custom_prompt_template
)

ov_model_path = f'./{model_name}/{model_precision}'
tokenizer = AutoTokenizer.from_pretrained(ov_model_path, trust_remote_code=True, padding=True, truncation=True, max_length=1024)
print(ov_model_path)
model = OVModelForCausalLM.from_pretrained(model_id=ov_model_path, device=inference_device, ov_config=ov_config)


def run_generation(text_user_en):
    
    vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    pipe = pipeline("text-generation", model=model, device='cpu', tokenizer=tokenizer, max_new_tokens=1024)

    llm = HuggingFacePipeline(
        pipeline=pipe, 
        model_kwargs={"temperature": 0.2}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=rag_chain_type,
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt}
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="refine", 
        retriever=retriever, 
        return_source_documents=False
    )
    
    #result = qa.run({"query": text_user_en})
    result = qa_chain.run(text_user_en)
    
    print("Full Text: ", result)
    matches = re.search(r'Answer:\s*([\s\S]*)', result)

    print("Matched: ",matches.group(1))
    return matches.group(1)

    complete_answers = [match.strip() for match in matches if len(match.strip()) > 20]
    if complete_answers:
       return complete_answers
    return matches





#app = FastAPI()

#@app.get('/chatbot/{item_id}')
def root(query:str=None):

    stime = time.time()
    ans = run_generation(query)
    etime = time.time()
    wc = len(ans.split())  # simple word count
    process_time = etime - stime
    words_per_sec = wc / process_time
    #return JSONResponse(content={
    #    'response': f'{ans} \r\n\r\nWord count: {wc}, Processing Time: {process_time:6.1f} sec, {words_per_sec:6.2} words/sec'
    #})
    #print("Responce: ", ans)
    return ans


