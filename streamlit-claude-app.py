import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Anthropic
import faiss
import re
import anthropic
from collections import Counter
import altair as alt

# Function to create config.toml dynamically
def create_streamlit_config():
    # Define the config path inside the app's directory
    config_dir = os.path.join(os.getcwd(), ".streamlit")
    config_path = os.path.join(config_dir, "config.toml")

    # Ensure the .streamlit directory exists
    os.makedirs(config_dir, exist_ok=True)

    # TOML content for Streamlit theme
    config_content = """ 
    [theme]
    base="dark"
    primaryColor="#5246d6"
    backgroundColor="#121212"
    secondaryBackgroundColor="#1E1E1E"
    textColor="#FFFFFF"
    font="monospace"
    """

    # Write the config file
    with open(config_path, "w") as config_file:
        config_file.write(config_content)

# Create the config before running the app
create_streamlit_config()

# Function to extract years from text
def extract_years(text):
    # Pattern to match years between 1900 and current year
    year_pattern = r'\b(19[0-9][0-9]|20[0-2][0-9])\b'
    years = re.findall(year_pattern, text)
    return [int(year) for year in years]

# Function to extract publication details using Claude
def extract_publication_details(client, text_chunks, num_publications=25):
    publications = []
    
    # Combine some chunks for context (adjust as needed)
    combined_text = " ".join(text_chunks[:10])
    
    prompt = f"""
    Based on the following academic text, extract information about {num_publications} most valuable publications mentioned.
    For each publication, provide:
    1. Title
    2. Authors
    3. Year
    4. Methodology
    5. Data used
    6. Key findings
    
    Format as a table with these columns. If any information is not available, write "N/A".
    
    TEXT:
    {combined_text}
    """
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Parse the table from response
    result = response.content[0].text
    
    # You'll need to parse the table format from Claude's response
    # This is a simplified approach - you might need to improve it
    lines = result.strip().split('\n')
    headers = []
    
    for i, line in enumerate(lines):
        if '|' in line:
            if not headers:
                headers = [h.strip() for h in line.split('|')[1:-1]]
            else:
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if len(cells) == len(headers) and not all(c.startswith('-') for c in cells):
                    pub = {headers[j]: cells[j] for j in range(len(headers))}
                    publications.append(pub)
    
    return publications

# Function to classify studies by methodology
def classify_by_methodology(client, text_chunks):
    # Combine some chunks for context
    combined_text = " ".join(text_chunks[:15])
    
    prompt = f"""
    Based on the following academic text, classify the studies mentioned by methodology.
    Create a summary of how many papers used each methodology (e.g., 60 papers used surveys, 25 used experiments, etc.)
    
    TEXT:
    {combined_text}
    """
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Parse the response to extract methodology counts
    result = response.content[0].text
    
    # This is a simplified approach - you might need custom parsing logic
    methodologies = {}
    lines = result.strip().split('\n')
    
    for line in lines:
        if ':' in line:
            parts = line.split(':')
            if len(parts) == 2:
                method = parts[0].strip()
                count_match = re.search(r'\d+', parts[1])
                if count_match:
                    count = int(count_match.group())
                    methodologies[method] = count
    
    return methodologies

# Streamlit App
def main():
    st.header('🌿 AI agent for Doctoral Students: Chat with Academic Papers 💬')
    st.sidebar.title('📚 LLM ChatApp using Claude & LangChain')
    
    key = st.text_input("Insert your Anthropic API key", type="password")
    
    if key:
        os.environ['ANTHROPIC_API_KEY'] = key
        client = anthropic.Anthropic(api_key=key)
        
        # Add some information about the app
        st.sidebar.markdown('''
        This app helps doctoral students analyze academic papers with:
        - Question answering about paper content
        - Publication year trends visualization
        - Summary of valuable publications
        - Classification of research methodologies
        ''')
        
        # File upload
        pdf = st.file_uploader("Upload your Paper (PDF)", type='pdf')
        
        if pdf is not None:
            with st.spinner("Processing your PDF..."):
                # Read PDF
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text=text)
                
                # Store name based on PDF name
                store_name = pdf.name[:-4]
                st.write(f"Processing: {store_name}")
                
                # Use HuggingFace embeddings since we're not using OpenAI
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # Check if embeddings exist, otherwise create them
                if os.path.exists(f"{store_name}.faiss"):
                    vector_store = FAISS.load_local(store_name, embeddings)
                    st.success('Embeddings Loaded from FAISS file')
                else:
                    vector_store = FAISS.from_texts(chunks, embeddings)
                    vector_store.save_local(store_name)
                    st.success('Embeddings Created and Saved')
                
                # Create tabs for different functionalities
                tab1, tab2, tab3, tab4 = st.tabs(["Chat with PDF", "Publication Years", "Top Publications", "Methodology Analysis"])
                
                with tab1:
                    st.subheader("Ask questions about the paper")
                    query = st.text_input("Ask a question about your PDF")
                    
                    if query:
                        with st.spinner("Generating answer..."):
                            # Get relevant documents
                            docs = vector_store.similarity_search(query=query, k=3)
                            
                            # Extract content from docs
                            doc_content = "\n".join([doc.page_content for doc in docs])
                            
                            # Use Claude to answer
                            prompt = f"""
                            Based on the following content from an academic paper, please answer this question:
                            
                            Question: {query}
                            
                            Content:
                            {doc_content}
                            """
                            
                            response = client.messages.create(
                                model="claude-3-5-sonnet-20240620",
                                max_tokens=1500,
                                messages=[
                                    {"role": "user", "content": prompt}
                                ]
                            )
                            
                            st.write(response.content[0].text)
                
                with tab2:
                    st.subheader("Publications by Year")
                    with st.spinner("Analyzing publication years..."):
                        # Extract years from the text
                        years = extract_years(text)
                        
                        if years:
                            # Count occurrences of each year
                            year_counts = Counter(years)
                            
                            # Convert to DataFrame for plotting
                            year_df = pd.DataFrame({
                                'Year': list(year_counts.keys()),
                                'Count': list(year_counts.values())
                            }).sort_values('Year')
                            
                            # Create Altair chart
                            chart = alt.Chart(year_df).mark_bar().encode(
                                x=alt.X('Year:O', sort=None),
                                y='Count:Q',
                                color=alt.Color('Count:Q', scale=alt.Scale(scheme='viridis')),
                                tooltip=['Year', 'Count']
                            ).properties(
                                title='Number of Publications by Year',
                                width=600,
                                height=400
                            )
                            
                            st.altair_chart(chart, use_container_width=True)
                            
                            # Add some analysis
                            most_common_year = year_df.loc[year_df['Count'].idxmax(), 'Year']
                            st.write(f"The topic appears most frequently in publications from {most_common_year}.")
                            
                            # Recent trend
                            recent_years = year_df[year_df['Year'] >= 2015]
                            if not recent_years.empty:
                                recent_count = recent_years['Count'].sum()
                                total_count = year_df['Count'].sum()
                                recent_percentage = (recent_count / total_count) * 100
                                
                                st.write(f"Publications from 2015 onwards represent {recent_percentage:.1f}% of all references.")
                        else:
                            st.warning("No publication years detected in the document.")
                
                with tab3:
                    st.subheader("Top Publications Analysis")
                    num_pubs = st.slider("Number of publications to extract", 5, 30, 20)
                    
                    if st.button("Extract Publications"):
                        with st.spinner("Analyzing publications..."):
                            publications = extract_publication_details(client, chunks, num_pubs)
                            
                            if publications:
                                # Convert to DataFrame
                                pub_df = pd.DataFrame(publications)
                                st.dataframe(pub_df, use_container_width=True)
                                
                                # Option to download as CSV
                                csv = pub_df.to_csv(index=False)
                                st.download_button(
                                    label="Download as CSV",
                                    data=csv,
                                    file_name=f"{store_name}_publications.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning("Could not extract publication details. Try with a different document.")
                
                with tab4:
                    st.subheader("Methodology Classification")
                    
                    if st.button("Analyze Methodologies"):
                        with st.spinner("Classifying methodologies..."):
                            methodologies = classify_by_methodology(client, chunks)
                            
                            if methodologies:
                                # Convert to DataFrame
                                method_df = pd.DataFrame({
                                    'Methodology': list(methodologies.keys()),
                                    'Count': list(methodologies.values())
                                }).sort_values('Count', ascending=False)
                                
                                # Display as table
                                st.dataframe(method_df, use_container_width=True)
                                
                                # Create pie chart
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.pie(method_df['Count'], labels=method_df['Methodology'], autopct='%1.1f%%', 
                                      startangle=90, shadow=True)
                                ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
                                
                                st.pyplot(fig)
                                
                                # Option to download as CSV
                                csv = method_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Methodology Data",
                                    data=csv,
                                    file_name=f"{store_name}_methodologies.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning("Could not classify methodologies. Try with a different document.")
    else:
        st.warning("Please enter your Anthropic API key to use this application.")

if __name__ == '__main__':
    main()
