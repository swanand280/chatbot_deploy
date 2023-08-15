import pickle
from flask import Flask, render_template, request, jsonify
import os
from flask_cors import CORS

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

app=Flask(__name__)
CORS(app,origins="*")

#Deserialise / Depickle
with open("faiss_store_openai.pkl", "rb") as f:
    vectorstore = pickle.load(f)

@app.route('/check',methods=['POST','GET'])
def predict_class():
    data = request.get_json()
    question = data.get("userInput")

    #features=[x for x in request.form.values()]
    #print(features[0])

    from langchain.chains import RetrievalQAWithSourcesChain
    from langchain.chains.question_answering import load_qa_chain
    from langchain import OpenAI
    llm = OpenAI(temperature=0)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

    if question:
        output = chain({"question": question}, return_only_outputs=True)
        print(output.get('answer'))
        #render_template("index.html",check=output.get('answer'))
        sudh = jsonify(output.get('answer'))
        print(sudh)
        return sudh
    else:
        return print("Error") #("index.html",check="Please Enter Data !")

if __name__ == "__main__":
    app.run(debug=True) #create a flask local server
