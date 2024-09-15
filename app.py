import os
from langchain.document_loaders import WikipediaLoader
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_groq import ChatGroq

from langchain_experimental.graph_transformers import LLMGraphTransformer

from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Neo4jVector
from typing import Tuple, List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import RunnableBranch,RunnableLambda,RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser









os.environ['GROQ_API_KEY']=
os.environ["NEO4J_URI"]=
os.environ["NEO4J_USERNAME"]=
os.environ["NEO4J_PASSWORD"]=

#initliazing Graph
graph=Neo4jGraph()

#loading docs from wikipidea
raw_docs=WikipediaLoader(query="Elizabath I ").load()


#splitting odcs so it can fit the llm context window
text_splitter=TokenTextSplitter(chunk_size=512,chunk_overlap=24)
documents=text_splitter.split_documents(raw_docs[:3])


#initializing chat grop llm 
llm=ChatGroq(groq_api_key="",model_name="Gemma2-9b-It")


#Graph LLm to find relatsion ship between the etities
llm_transformer=LLMGraphTransformer(llm=llm)


#converting all the docs to graph
graph_docs=llm_transformer.convert_to_graph_documents(documents)

#intilaing my graph fro my data set 
graph.add_graph_documents(
    graph_docs,
    baseEntityLabel=True,
    include_source=True

)

#default cipher langauge to find entites -> relatsionship
default_cipher="MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"



#showing Graph displaying graph

def showGraph(cypher:str=default_cipher):
  driver=GraphDatabase.driver(
      uri=os.environ["NEO4J_URI"],auth=(os.environ["NEO4J_USERNAME"],os.environ["NEO4J_PASSWORD"])

  )
  session=driver.session()
  widget=GraphWidget(graph=session.run(cypher).graph())
  widget.node_label_mapping='id'
  display(widget)
  return widget



#normal vector embedding it will be used with graph db 
vector_index=Neo4jVector.from_existing_graph(
    HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"




)

#finding id from the graph
graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")


# for getting the desie output in the variable
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )


#inttilaizng the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)


#1st chain
entity_chain = prompt | llm.with_structured_output(Entities)


#invoking cain
entity_chain.invoke({"question": "Where was Amelia Earhart born?"}).names

#geeting tall the relations hips from the graph about our data 
def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# structuerd output getting out all the entities related to te node
def structured_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit: 2})
            YIELD node, score
            WITH node
            CALL {
                MATCH (node)-[r:MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                MATCH (node)<-[r:MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result



#combination of structured adn unstructured retriever

def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    return final_data




# templaet to answer the question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""


CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x : x["question"]),
)




template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""







prompt = ChatPromptTemplate.from_template(template)




#chain2 
chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)


#question1
chain.invoke({"question": "Which house did Elizabeth I belong to?"})
# answer:  \n\nThis text is about Stephen Richard Lowe, a British bishop. \n'




#question2
chain.invoke(
    {
        "question": "When was she born?",
        "chat_history": [("Which house did Elizabeth I belong to?", "House Of Tudor")],
    }
)
#answer : 'March 3, 1944 \n' the chat also knows the answer to the previosn questioon




