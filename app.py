import streamlit as st
import pandas as pd
from classifier import NAICSClassifier
import os
from time import sleep
import asyncio

# Page configuration
st.set_page_config(page_title="NAICS Classifier", layout="wide")
st.title("NAICS Code Classification Tool")

# Session state initialization
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'debug' not in st.session_state:
    st.session_state.debug = False
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0
if 'total_processed' not in st.session_state:
    st.session_state.total_processed = 0
if 'progress_bar' not in st.session_state:
    st.session_state.progress_bar = None
if 'status_text' not in st.session_state:
    st.session_state.status_text = None
if 'results' not in st.session_state:
    st.session_state.results = []

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    perplexity_api_key = st.text_input("Perplexity API Key", type="password")
    st.session_state.debug = st.checkbox("Enable Debug Mode")
    if api_key and api_key.startswith('sk-'):
        try:
            if not perplexity_api_key:
                st.warning("Perplexity API key is required for company research")
            else:
                st.session_state.classifier = NAICSClassifier(api_key, perplexity_api_key)
                st.success("API keys validated")
        except Exception as e:
            st.error(f"Invalid API key: {str(e)}")
    st.info("Enter your API keys to enable classification")

# File validation function
def validate_files(naics_df, company_df):
    required_naics_cols = ['2022 NAICS Code', '2022 NAICS Title']
    required_company_cols = ['Company']
    
    if not all(col in naics_df.columns for col in required_naics_cols):
        st.error("Missing required columns in NAICS file")
        return False
    
    if not all(col in company_df.columns for col in required_company_cols):
        st.error("Missing required columns in Company file")
        return False
        
    return True

# Main form with improved layout
with st.form("naics_classifier"):
    st.header("Upload Files")
    
    col1, col2 = st.columns(2)
    with col1:
        naics_file = st.file_uploader("NAICS Code File (Excel)", type=["xlsx"])
    with col2:
        company_file = st.file_uploader("Company Directory File (Excel)", type=["xlsx"])
    
    submit_button = st.form_submit_button("Process Files")
    
    if submit_button and api_key and naics_file and company_file:
        try:
            # Validate files
            if naics_file.size == 0 or company_file.size == 0:
                st.error("Uploaded file is empty")
                st.stop()
                
            # Load and validate data
            naics_df = pd.read_excel(naics_file)[['2022 NAICS Code', '2022 NAICS Title']]
            company_df = pd.read_excel(company_file)
            
            if not validate_files(naics_df, company_df):
                st.stop()
            
            # Debug information
            if st.session_state.debug:
                st.write("NAICS DF Shape:", naics_df.shape)
                st.write("Company DF Columns:", list(company_df.columns))
            
            # Reset counters
            st.session_state.error_count = 0
            st.session_state.total_processed = 0
            st.session_state.results = []
            
            # Create progress bar once
            progress_bar = st.progress(0)
            status_text = st.empty()
            error_text = st.empty()
            
            total = len(company_df)
            for index in range(total):
                row = company_df.iloc[index]
                try:
                    progress = (index + 1) / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {index+1}/{total}: {row['Company']}")
                    
                    result = asyncio.run(st.session_state.classifier.process_company(row, naics_df))
                    st.session_state.results.append(result)
                    st.session_state.total_processed += 1
                    
                except Exception as e:
                    st.session_state.results.append((None, None, 0.0, f"Error: {str(e)}"))
                    st.session_state.error_count += 1
                    error_text.error(f"Errors: {st.session_state.error_count}/{index+1} companies")
                
                # Update status
                status_color = "🟡" if st.session_state.error_count > 0 else "🟢"
                status_text.markdown(f"{status_color} Processed: {st.session_state.total_processed}/{total} | Errors: {st.session_state.error_count}")
                
                sleep(0.1)  # Adjust delay as needed

            # Finalize processing: update dataframe and display results
            for i, result in enumerate(st.session_state.results):
                company_df.loc[i, ['NAICS Code', 'NAICS Description', 'Confidence', 'Source Method']] = result
            st.session_state.processed_data = company_df
            
            # Final status update
            if st.session_state.error_count > 0:
                error_text.error(f"Completed with {st.session_state.error_count} errors. Check results for details.")
            else:
                error_text.success("Processing complete with no errors!")

            # Optionally, clear the progress bar if you wish
            progress_bar.empty()
            st.session_state.current_index = 0  # Reset current_index

        except Exception as e:
            st.error(f"Critical error processing files: {str(e)}")
            if 'progress_bar' in locals():
                progress_bar.progress(1.0)
                status_text.error("Processing failed - see error above")

# Display results
if st.session_state.processed_data is not None:
    st.header("Results")
    st.dataframe(st.session_state.processed_data, use_container_width=True)
    
    csv = st.session_state.processed_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Processed Data",
        data=csv,
        file_name="processed_companies.csv",
        mime="text/csv"
    )
