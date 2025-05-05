import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
import requests
from typing import List, Dict
import json


load_dotenv()

st.set_page_config(
    page_title="Personal Research Assistant Agent",
    page_icon="🤖",
    layout="wide"
)

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

client = Groq(api_key=groq_api_key)

def search_papers(query: str, limit: int ) -> List[Dict]:
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,abstract,url,year,citationCount"
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json().get("data", [])
    return []

def summarize_papers(papers: List[Dict]) -> str:
    """Summarize papers using Groq LLM"""
    papers_text = "\n\n".join([
        f"Title: {paper['title']}\n"
        f"Authors: {', '.join([author['name'] for author in paper.get('authors', [])])}\n"
        f"Year: {paper.get('year', 'N/A')}\n"
        f"Abstract: {paper.get('abstract', 'No abstract available')}\n"
        f"Citation Count: {paper.get('citationCount', 0)}\n"
        f"URL: {paper.get('url', 'No URL available')}"
        for paper in papers
    ])
    
    prompt = f"""Please provide a comprehensive summary of the following research papers. 
    Focus on the key findings, methodologies, and contributions of each paper. 
    Also highlight any common themes or differences between the papers.
    
    Papers:
    {papers_text}
    
    Summary:"""
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.7,
        max_tokens=2000
    )
    
    return response.choices[0].message.content

st.title("Personal Research Assistant Agent 🤖")
st.write("This is a personal research assistant agent that can help you with your research.")


research_query = st.text_input("Enter your research query:")
number_of_papers = st.number_input("Enter the number of papers you want to search for:", min_value=1, max_value=10, value=5)

if research_query:
    with st.spinner("Searching for relevant papers..."):
        papers = search_papers(research_query, number_of_papers)
        
        if papers:
            st.success(f"Found {len(papers)} relevant papers!")
            
            with st.spinner("Summarizing papers..."):
                summary = summarize_papers(papers)
                
                st.subheader("Research Summary")
                st.write(summary)
                
                st.subheader("Source Papers")
                for paper in papers:
                    with st.expander(f"{paper['title']} ({paper.get('year', 'N/A')})"):
                        st.write(f"**Authors:** {', '.join([author['name'] for author in paper.get('authors', [])])}")
                        st.write(f"**Abstract:** {paper.get('abstract', 'No abstract available')}")
                        st.write(f"**Citation Count:** {paper.get('citationCount', 0)}")
                        st.write(f"**URL:** {paper.get('url', 'No URL available')}")
        else:
            st.error("No papers found for your query. Please try a different search term.")
