import streamlit as st
import os
import pandas as pd
import numpy as np
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
def extract_publication_details(client, text, num_publications=25):
    # Use a shorter text if it's too long
    if len(text) > 10000:
        text = text[:10000]
    
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
    {text}
    """
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse the table from response
        result = response.content[0].text
        
        # Create a list to store the publications
        publications = []
        
        # Simple parsing based on markdown table format
        lines = result.strip().split('\n')
        headers = []
        
        for i, line in enumerate(lines):
            if '|' in line:
                # Extract header row
                if not headers and i < len(lines) - 1 and '---' in lines[i+1]:
                    headers = [h.strip() for h in line.split('|') if h.strip()]
                # Extract data rows
                elif headers and not line.strip().startswith('|---'):
                    cells = [c.strip() for c in line.split('|') if c.strip()]
                    if len(cells) >= len(headers):
                        pub = {headers[j]: cells[j] for j in range(len(headers))}
                        publications.append(pub)
        
        return publications
    except Exception as e:
        st.error(f"Error extracting publications: {e}")
        return []

# Function to classify studies by methodology
def classify_by_methodology(client, text):
    # Use a shorter text if it's too long
    if len(text) > 10000:
        text = text[:10000]
    
    prompt = f"""
    Based on the following academic text, classify the studies mentioned by methodology.
    Create a summary of how many papers used each methodology (e.g., 60 papers used surveys, 25 used experiments, etc.)
    
    TEXT:
    {text}
    """
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse the response to extract methodology counts
        result = response.content[0].text
        
        # This is a simplified approach - assuming Claude returns a list
        methodologies = {}
        lines = result.strip().split('\n')
        
        for line in lines:
            # Look for patterns like "X papers used Y" or "Y: X papers"
            match1 = re.search(r'(\d+)\s+(?:papers|studies)\s+(?:used|employed|utilized)\s+([^:.,]+)', line)
            match2 = re.search(r'([^:.,]+):\s*(\d+)', line)
            
            if match1:
                count = int(match1.group(1))
                method = match1.group(2).strip()
                methodologies[method] = count
            elif match2:
                method = match2.group(1).strip()
                count = int(match2.group(2))
                methodologies[method] = count
        
        return methodologies
    except Exception as e:
        st.error(f"Error classifying methodologies: {e}")
        return {}

# Streamlit App
def main():
    st.header('🌿 AI agent for Doctoral Students: Chat with Academic Papers 💬')
    st.sidebar.title('📚 LLM ChatApp using Claude API')
    
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
                    page_text = page.extract_text()
                    if page_text:  # Check if text extraction was successful
                        text += page_text
                
                if not text:
                    st.error("Could not extract text from the PDF. Please try a different file.")
                    return
                
                # Store name based on PDF name
                store_name = pdf.name[:-4]
                st.write(f"Processing: {store_name}")
                
                # Create tabs for different functionalities
                tab1, tab2, tab3, tab4 = st.tabs(["Chat with PDF", "Publication Years", "Top Publications", "Methodology Analysis"])
                
                with tab1:
                    st.subheader("Ask questions about the paper")
                    query = st.text_input("Ask a question about your PDF")
                    
                    if query:
                        with st.spinner("Generating answer..."):
                            # Use Claude to answer directly from the text
                            prompt = f"""
                            Based on the following content from an academic paper, please answer this question:
                            
                            Question: {query}
                            
                            Content:
                            {text[:10000]}  # Using first 10,000 chars to stay within context limits
                            """
                            
                            try:
                                response = client.messages.create(
                                    model="claude-3-5-sonnet-20240620",
                                    max_tokens=1500,
                                    messages=[
                                        {"role": "user", "content": prompt}
                                    ]
                                )
                                
                                st.write(response.content[0].text)
                            except Exception as e:
                                st.error(f"Error getting answer: {e}")
                
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
                            publications = extract_publication_details(client, text, num_pubs)
                            
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
                                st.warning("Could not extract publication details. Try a different document or check your API key.")
                
                with tab4:
                    st.subheader("Methodology Classification")
                    
                    if st.button("Analyze Methodologies"):
                        with st.spinner("Classifying methodologies..."):
                            methodologies = classify_by_methodology(client, text)
                            
                            if methodologies:
                                # Convert to DataFrame
                                method_df = pd.DataFrame({
                                    'Methodology': list(methodologies.keys()),
                                    'Count': list(methodologies.values())
                                }).sort_values('Count', ascending=False)
                                
                                # Display as table
                                st.dataframe(method_df, use_container_width=True)
                                
                                # Create bar chart with Altair
                                chart = alt.Chart(method_df).mark_bar().encode(
                                    x='Count:Q',
                                    y=alt.Y('Methodology:N', sort='-x'),
                                    color=alt.Color('Count:Q', scale=alt.Scale(scheme='viridis')),
                                    tooltip=['Methodology', 'Count']
                                ).properties(
                                    title='Research Methodologies Used',
                                    width=600,
                                    height=400
                                )
                                
                                st.altair_chart(chart, use_container_width=True)
                                
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
