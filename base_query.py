import langchain, backoff, openai, requests, urllib.parse
import tkinter as tk
from tkinter import filedialog
import docx2txt
import PyPDF2
import pandas as pd
import pdfplumber
import warnings
warnings.filterwarnings('ignore')
from langchain.tools import HumanInputRun
#from apikeys import WolframAplha_KEY, SerpAPI_KEY
from langchain.agents import Tool
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
#from apikeys import OPENAI_KEY
import os
import re
import openai
import re
import math
import random
from tqdm import tqdm
import time
from openai import OpenAI

client = OpenAI(api_key = "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")

from langchain.graphs import Neo4jGraph

url = "neo4j+s://3719ef4f.databases.neo4j.io"
username ="neo4j"
password = "4rnhpVG8v-s-4iei7wGc0GaG2jBTSwAxYtrDiwOlqBo"
graph = Neo4jGraph(
    url=url,
    username=username,
    password=password
)

###Facts generation
def breakdown_answer(answer):
    # Split the input paragraph into sentences
    sentences = re.split(r'\.', answer)

    # Remove any empty strings resulting from split
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    # Now iterate over each sentence and send to OpenAI API
    for sentence in sentences:
        prompt = f"""You are a highly advanced language model programmed to decompose complex text into simpler factoids. Your primary task is to dissect larger narratives into standalone pieces of information, with each piece containing a single fact that can be independently verified.

    Your goal is to retain the original meaning and context of the information without making any corrections or alterations. This includes both explicit details and implied facts present in the original text. Additionally, you should be able to identify and illustrate dependencies between factoids, creating a hierarchy that reflects the nested structure of the facts.

    Even if the information appears incorrect, your task is not to correct it, but to reproduce it faithfully in a simpler, structured form that can be easily verified or refuted.
        Apply this to the following statement:
    {sentence}
    Output full sentences only which are grammatically correct. Each sentence should be in a new line. Each sentence should be factually independent. Ignore all conjunctive adverbs.
        """

        prompt1 = f"""You are a highly advanced language model programmed to decompose complex text into simpler factoids. Your primary task is to dissect larger narratives into standalone pieces of information, with each piece containing a single fact that can be independently verified.

    Your goal is to retain the original meaning and context of the information without making any corrections or alterations. This includes both explicit details and implied facts present in the original text. Additionally, you should be able to identify and illustrate dependencies between factoids, creating a hierarchy that reflects the nested structure of the facts.

    Even if the information appears incorrect, your task is not to correct it, but to reproduce it faithfully in a simpler, structured form that can be easily verified or refuted.
        """
        prompt2 = f"""
        Your goal is to retain the original meaning and context of the information without making any corrections or alterations. This includes both explicit details and implied facts present in the original text. Additionally, you should be able to identify and illustrate dependencies between factoids, creating a hierarchy that reflects the nested structure of the facts.

    Even if the information appears incorrect, your task is not to correct it, but to reproduce it faithfully in a simpler, structured form that can be easily verified or refuted.

    Apply this to the following statement:
    {sentence}
    Output only full sentences, each in a new line. Ignore all conjunctive adverbs.
        """


        response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": prompt1},
            {"role": "user", "content": prompt2}
        ],
        temperature=0,
        max_tokens=500
        )
        raw_chunks = response.choices[0].message.content.split("\n")
        for ele in raw_chunks:
          chunks.append(ele)
        # # Split raw chunks by full stop and comma
        # for chunk in raw_chunks:
        #     sub_chunks = re.split(r'[.,;]', chunk)
        #     # Filter out chunks that are purely numerical and add to the list
        #     chunks.extend([sub.strip() for sub in sub_chunks if sub.strip() and not sub.strip().isdigit()])

    return chunks

###Basic functions

from langchain_community.graphs.graph_document import (
    Node,
    Relationship,
    GraphDocument,
)
from langchain.schema import Document
from typing import List, Dict, Any, Optional
from langchain.pydantic_v1 import Field, BaseModel

class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""
    nodes: List[Node] = Field(
        ..., description="List of nodes in the knowledge graph")
    rels: List[Relationship] = Field(
        ..., description="List of relationships in the knowledge graph"
    )





def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)

def props_to_dict(props) -> dict:
    """Convert properties to a dictionary."""
    properties = {}
    if not props:
      return properties
    for p in props:
        properties[format_property_key(p.key)] = p.value
    return properties

def map_to_base_node(node: Node) -> Node:
    """Map the KnowledgeGraph Node to the base Node."""
    properties = props_to_dict(node.properties) if node.properties else {}
    # Add name property for better Cypher statement generation
    properties["name"] = node.id.title()
    return Node(
        id=node.id.title(), type=node.type.capitalize(), properties=properties
    )


def map_to_base_relationship(rel: Relationship) -> Relationship:
    """Map the KnowledgeGraph Relationship to the base Relationship."""
    source = map_to_base_node(rel.source)
    target = map_to_base_node(rel.target)
    properties = props_to_dict(rel.properties) if rel.properties else {}
    return Relationship(
        source=source, target=target, type=rel.type, properties=properties
    )




import os
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

def get_extraction_chain(llm,
    prompting,
    allowed_nodes: Optional[List[str]] = None,
    allowed_rels: Optional[List[str]] = None
    ):
    prompt = ChatPromptTemplate.from_messages(
        [(
          "system",prompting
),
            ("human",  "Use the given format to extract information from the following list of facts. If there are any dates or numbers, do not forget them in the nodes and relationships: {input}"),
            ("human","Tip: Make sure to answer in the correct format. Don't forget the numbers in your extraction of nodes and relationships. Include them as relationships, not proprieties"),
        ])
    return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=False)



def extract_relationships_with_nodes(nodes, rels):
    extracted_relationships = []
    for rel in rels:
        source_node_id = rel.source
        target_node_id = rel.target
        source_node = next((node for node in nodes if node.id == source_node_id.id), None)
        target_node = next((node for node in nodes if node.id == target_node_id.id), None)
        if source_node and target_node:
            source = Node(id=source_node.id, type=source_node.type, properties=source_node.properties)
            target = Node(id=target_node.id, type=target_node.type, properties=target_node.properties)
            relationship = Relationship(source=source, target=target, type=rel.type)
            if relationship not in extracted_relationships:
              extracted_relationships.append(relationship)
    return extracted_relationships



def extract_and_store_graph(document: Document,
    llm,
    prompting,
    facts,
    nodes:Optional[List[str]] = None,
    rels:Optional[List[str]]=None) -> None:
    extract_chain = get_extraction_chain(llm, prompting, nodes, rels)
    data = extract_chain.invoke(facts)['function']
    nodes = [map_to_base_node(node) for node in data.nodes]
    relations = extract_relationships_with_nodes(data.nodes, data.rels)
    graph_document = GraphDocument(
      nodes = [map_to_base_node(node) for node in data.nodes],
      relationships = extract_relationships_with_nodes(data.nodes, data.rels),
      source = document
    )
    print('_____________________________________', graph_document)
    # Store information into a graph
    graph.add_graph_documents([graph_document])
    return nodes, relations


from graphviz import Digraph
import random
from IPython.display import display

def generate_random_color():
    # Generate a random color in hexadecimal format
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def draw_knowledge_graph(nodes, relationships):
    dot = Digraph(comment="Knowledge Graph")

    # Define colors for different node types randomly
    color_map = {}
    rel_nodes = []
    for rel in relationships:
      rel_nodes.append(rel.source)
      rel_nodes.append(rel.target)
    for node in rel_nodes:
        if node.type not in color_map:
            color_map[node.type] = generate_random_color()

    # Add nodes to the graph with colors
    for node in rel_nodes:
        node_id = node.id
        node_label = f"{node.id}\n{node.type}"
        color = color_map.get(node.type, "black")  # Default to black if type not found
        dot.node(node_id, node_label, shape='rect', style='filled', fillcolor=color)

    # Add relationships as edges to the graph
    for relationship in relationships:
        dot.edge(relationship.source.id, relationship.target.id, label=relationship.type)

    # Ask the user for a name for the graph
    name = input("Enter a name for the graph: ")
    # Save the graph as a PDF under the user-provided name
    dot.render(name, format='pdf', cleanup=True)

    # Display the graph
    display(dot)

def my_graph(document, llm, my_prompting, my_facts):
    nodes, rels = extract_and_store_graph(document,llm, my_prompting, my_facts)
    draw_knowledge_graph(nodes, rels)




from langchain.chains import GraphCypherQAChain
def ask_question_to_KG(question):
   # Query the knowledge graph in a RAG application
    graph.refresh_schema()

    cypher_chain = GraphCypherQAChain.from_llm(
        graph=graph,
        cypher_llm=ChatOpenAI(temperature=0, model="gpt-4"),
        qa_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        validate_cypher=True, # Validate relationship directions
        verbose=True
    )
    answer = cypher_chain.invoke({"query": question})
    return answer['result']