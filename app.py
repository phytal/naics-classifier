import streamlit as st
import pandas as pd
from classifier import NAICSClassifier
import os
import asyncio
import atexit
import nest_asyncio
nest_asyncio.apply()

# Page configuration
st.set_page_config(page_title="NAICS Classifier", layout="wide")
st.title("NAICS Code Classification Tool")

def cleanup_resources():
    """Enhanced cleanup function"""
    try:
        # Clean up classifier resources
        if 'classifier' in st.session_state and st.session_state.classifier:
            try:
                del st.session_state.classifier.vector_index
                del st.session_state.classifier
            except:
                pass
        
        # Clean up any running event loops
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.stop()
            if not loop.is_closed():
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()
        except Exception:
            pass
        
        # Force garbage collection
        import gc
        gc.collect()
        
    except Exception as e:
        st.error(f"Error during cleanup: {e}")

# Register cleanup
atexit.register(cleanup_resources)

# Initialize session state for cleanup
if 'cleanup_registered' not in st.session_state:
    st.session_state.cleanup_registered = False
    
if not st.session_state.cleanup_registered:
    # Add cleanup to session state
    st.session_state.cleanup = cleanup_resources
    st.session_state.cleanup_registered = True

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
if 'naics_df' not in st.session_state:
    st.session_state.naics_df = None

class AsyncEventLoopHandler:
    """Handle async event loop lifecycle"""
    def __init__(self):
        self.loop = None
        
    def get_loop(self):
        if self.loop is None or self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        return self.loop
        
    def run_async(self, coro):
        loop = self.get_loop()
        try:
            return loop.run_until_complete(coro)
        except Exception as e:
            raise e

# Initialize event loop handler in session state
if 'event_loop_handler' not in st.session_state:
    st.session_state.event_loop_handler = AsyncEventLoopHandler()

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    perplexity_api_key = st.text_input("Perplexity API Key", type="password")
    st.session_state.debug = st.checkbox("Enable Debug Mode")
    
    # Remove classifier initialization from sidebar
    if not api_key or not api_key.startswith('sk-'):
        st.warning("Please enter a valid OpenAI API key")
    elif not perplexity_api_key:
        st.warning("Perplexity API key is required for company research")
    
    st.info("Upload NAICS file and enter API keys to enable classification")

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
            
            # Initialize classifier with NAICS data
            try:
                st.session_state.classifier = NAICSClassifier(api_key, perplexity_api_key, naics_df)
                st.success("Classifier initialized successfully")
            except Exception as e:
                st.error(f"Failed to initialize classifier: {str(e)}")
                st.stop()
            
            # Debug information
            if st.session_state.debug:
                st.write("NAICS DF Shape:", naics_df.shape)
                st.write("Company DF Columns:", list(company_df.columns))
            
            # Store naics_df in session state
            st.session_state.naics_df = naics_df
            
            # Verify classifier is initialized
            if st.session_state.classifier is None:
                st.error("Classifier not initialized. Please check API keys and try again.")
                st.stop()
            
            # Reset counters
            st.session_state.error_count = 0
            st.session_state.total_processed = 0
            st.session_state.results = []
            
            # Create progress bar once
            progress_bar = st.progress(0)
            status_text = st.empty()
            error_text = st.empty()
            
            total = len(company_df)

            # Define progress update function
            def update_progress(current, total, company, success):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"Processing {current}/{total}: {company}")
                if not success:
                    st.session_state.error_count += 1
                    error_text.error(f"Errors: {st.session_state.error_count}/{current} companies")
                st.session_state.total_processed = current
                
                # Update status
                status_color = "🟡" if st.session_state.error_count > 0 else "🟢"
                status_text.markdown(f"{status_color} Processed: {current}/{total} | Errors: {st.session_state.error_count}")

            # Modify the processing loop to use event loop handler
            async def process_companies(classifier, company_df, naics_df, progress_callback):
                """Process companies using managed event loop"""
                results = []
                for index in range(len(company_df)):
                    try:
                        row = company_df.iloc[index]
                        result = await classifier.process_company(row, naics_df)
                        results.append(result)
                        progress_callback(index + 1, len(company_df), row['Company'], True)
                    except Exception as e:
                        results.append((None, None, 0.0, f"Error: {str(e)}"))
                        progress_callback(index + 1, len(company_df), row['Company'], False)
                return results

            try:
                # Process companies using managed event loop
                st.session_state.results = st.session_state.event_loop_handler.run_async(
                    process_companies(
                        st.session_state.classifier,
                        company_df,
                        naics_df,
                        update_progress
                    )
                )
                
                # Rest of processing logic
                for i, result in enumerate(st.session_state.results):
                    naics_code, description, confidence, source = result
                    
                    # If description is None but we have a valid code, look up the description
                    if description is None and naics_code is not None:
                        # Convert NAICS codes to strings for comparison
                        naics_code_str = str(naics_code)
                        matching_desc = naics_df[naics_df['2022 NAICS Code'].astype(str) == naics_code_str]['2022 NAICS Title'].iloc[0] if any(naics_df['2022 NAICS Code'].astype(str) == naics_code_str) else None
                        description = matching_desc
                    
                    company_df.loc[i, ['NAICS Code', 'NAICS Description', 'Confidence', 'Source Method']] = [
                        naics_code,
                        description,
                        confidence,
                        source
                    ]
                
                st.session_state.processed_data = company_df
                
                # Final status update
                if st.session_state.error_count > 0:
                    error_text.error(f"Completed with {st.session_state.error_count} errors. Check results for details.")
                else:
                    error_text.success("Processing complete with no errors!")

                # Optionally, clear the progress bar if you wish
                progress_bar.empty()
                st.session_state.current_index = 0  # Reset current_index

                # Ensure cleanup after processing
                if hasattr(st.session_state, 'cleanup'):
                    st.session_state.cleanup()

            except Exception as e:
                st.error(f"Processing error: {str(e)}")
                if hasattr(st.session_state, 'cleanup'):
                    st.session_state.cleanup()
            finally:
                # Cleanup
                if hasattr(st.session_state, 'event_loop_handler'):
                    try:
                        loop = st.session_state.event_loop_handler.loop
                        if loop and not loop.is_closed():
                            loop.close()
                    except:
                        pass

        except Exception as e:
            if hasattr(st.session_state, 'cleanup'):
                st.session_state.cleanup()
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
