import azure.functions as func
import logging

from langchain.chains import RetrievalQA
from azure_search import create_azure_search

from llm import create_llm

def create_vector_store():
    return create_azure_search()

logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

#ques ="What is the scope of the proposal for the Battery Storage system in Rauma?"
#ques = "What is the proposal about?" 
#ques = "What kind of expertise is needed in the project?"
#ques = "Make a list of all deliverables in each part of the proposal"
#ques = "Generate battery energy storage system preliminary design proposal topics based on this document and add more if needed"
#print(res['result'])


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
@app.function_name(name="energy_copilot_test")
@app.route(route="energy_copilot_test")
def energy_copilot_test(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')


    db = create_vector_store()
    llm = create_llm()


    retriever=db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 50, 'score_threshold':0.8}
    )

    qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever,chain_type="stuff")


    ques = req.params.get('ques')
    if not ques:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            ques = req_body.get('ques')

    if ques:
        result = qa_chain({"query": ques})
        res = result['result']
        return func.HttpResponse(f"{res}. This HTTP triggered function executed successfully.")

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")

    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )

