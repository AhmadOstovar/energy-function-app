#%%
import logging
from typing import List, Tuple
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from azure_search import create_azure_search
from embeddings import create_embeddings
import langchain
from langchain.docstore.document import Document
from langchain.retrievers import AzureCognitiveSearchRetriever

from llm import create_llm

#%%
def create_vector_store():
    return create_azure_search()
#    return FAISS.load_local("faiss_index", create_embeddings())

def print_docs(docs:List[Document]):
    for doc in docs:
        print(doc.page_content)

logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)
#langchain.verbose = True

db = create_vector_store()
llm = create_llm()


retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 50, 'score_threshold':0.8}
)

qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever,chain_type="stuff")

#%%
# Helper function for printing docs

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

#%%
#ques ="What is the scope of the proposal for the Battery Storage system in Rauma?"
#ques = "What is the proposal about?" 
#ques = "What kind of expertise is needed in the project?"
#ques = "Make a list of all deliverables in each part of the proposal"
ques = "Generate battery energy storage system preliminary design proposal topics based on this document and add more if needed"
res = qa_chain({"query": ques})
print(res['result'])

'''
docs = retriever.get_relevant_documents(query=ques)
for doc in docs:
    print(doc.page_content)
    print(doc.metadata['source'])

'''
