# LLM-Enhanced Knowledge Graph Construction & Question Answering

This repository explores various strategies for building Knowledge Graphs (KGs) using Large Language Models (LLMs), optimizing their performance in Question Answering (QA) tasks, and evaluating the impact of different techniques such as few-shot prompting and fine-tuning.

## Table of Contents
1. [Introduction](#introduction)
2. [KG Construction Approaches](#kg-construction-approaches)
   - [Contextual KG Construction](#contextual-kg-construction)
   - [Chunking and Fact Extraction](#chunking-and-fact-extraction)
3. [Effect of Few-Shot Prompting](#effect-of-few-shot-prompting)
4. [Fine-Tuning for KG Formatting](#fine-tuning-for-kg-formatting)
5. [Vector Databases vs KGs for QA](#vector-databases-vs-kgs-for-qa)
6. [Source Tracking in QA](#source-tracking-in-qa)
7. [How to Use](#how-to-use)
8. [Installation](#installation)
9. [Contributions](#contributions)
10. [License](#license)

---

## Introduction
In this project, we investigate methods for leveraging LLMs to construct Knowledge Graphs (KGs) and perform accurate Question Answering (QA). The goal is to refine KG extraction techniques while evaluating the trade-offs between KGs and vector databases for storing and querying knowledge. We also explore approaches to track the origin of facts within the context used for KG creation.

## KG Construction Approaches

### Contextual KG Construction: Chunking and Fact Extraction
This section details the normal approach for extracting facts from the entire context using LLMs to build KGs. Before the extraction, we go through a first pre-processing step: chunking and fact extraction. The context is divided into chunks and facts are extracted individually from each chunk before being integrated into the KG. This method aims to improve precision by focusing on smaller, more manageable text units.

We use the `breakdown_answer` function to divide our text into a list of facts.  Then we use the following prompt to extract the Knowledge Graph:
```bash
prompt = ChatPromptTemplate.from_messages(
        [(
          "system",prompting
),
            ("human",  "Use the given format to extract information from the following list of facts. If there are any dates or numbers, do not forget them in the nodes and relationships: {input}"),
            ("human","Tip: Make sure to answer in the correct format. Don't forget the numbers in your extraction of nodes and relationships. Include them as relationships, not proprieties"),
        ])
```
The Knowledge Graph's extraction and the facts extraction's codes are in the `base_query.py` file. `Querying.ipynb` contains an example of how to run the KG extraction and how to query it for answers. 

<p align="center">
  <img src="images/architecture.png" alt="Architectures Diagram" width="90%" />
</p>

## Effect of Few-Shot Prompting
Few-shot prompting is critical in improving the precision of KG extraction. This section analyzes the effect of providing relevant examples in the prompt to guide the LLM's fact extraction process. Experiments compare results with and without few-shot prompting to assess the impact on KG accuracy.

## Fine-Tuning for KG Formatting
LLMs not predisposed to outputting facts in a structured KG format often require fine-tuning. Here, we detail the methods used to fine-tune models so they can generate well-structured KGs. This includes adjusting the LLM's output style to match KG requirements and formatting.

## Vector Databases vs KGs for QA
This section explores the trade-offs between using vector databases and traditional KGs for answering questions. We compare their respective advantages in terms of retrieval speed, answer accuracy, and scalability when used for QA tasks.

## Source Tracking in QA
In addition to providing answers, we also implement a method for tracing the paragraph or section of the original context that provided the extracted information. This bonus feature enhances transparency by linking each fact to its source within the input text.

## How to Use
[Instructions on how to interact with the repository’s code and models for KG construction and QA.]

(Add any specific usage examples here.)

## Installation
[Provide installation instructions, including any dependencies.]

```bash
# Example installation command
pip install -r requirements.txt
