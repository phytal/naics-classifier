from openai import OpenAI
import re
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from typing import Optional, Tuple, List, Dict
import pandas as pd
from utils import rate_limited, throttled
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import ModelRetry
import os
import logging
from vector_index import NAICSVectorIndex, create_naics_search_tool
import asyncio

class NAICSDependencies(BaseModel):
    """Dependencies for NAICS classification"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    vector_index: NAICSVectorIndex = Field(..., description="FAISS vector index for NAICS search")
    company_info: str = Field(..., description="Company information to classify")

class NAICSResponse(BaseModel):
    code: str = Field(..., pattern=r"^\d{6}$", description="Primary NAICS Code")
    industries: List[str] = Field(..., min_items=1, description="Matched industry categories")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence Percentage")
    reason: str = Field(..., max_length=280, description="Classification rationale")
    candidates: List[Dict] = Field(..., description="Vector search results")

class ClassificationError(Exception):
    def __init__(self, message: str, candidates: Optional[List[Dict]] = None):
        super().__init__(message)
        self.candidates = candidates

class NAICSClassifier:
    def __init__(self, openai_api_key: str, perplexity_api_key: str, naics_df: pd.DataFrame):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.perplexity_client = OpenAI(
            api_key=perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )
        self.test_connection()
        self.consecutive_errors = 0
        self.vector_index = NAICSVectorIndex(naics_df)
        
        # Create agent with improved retry configuration
        self.react_agent = Agent(
            'openai:gpt-4o',
            deps_type=NAICSDependencies,
            result_type=NAICSResponse,
            system_prompt="""
You are an expert at NAICS classification. Follow these rules:
1. Always return valid 6-digit NAICS codes
2. If you have insufficient information about the company or are uncertain, return code 999999
3. Format codes as strings like "541511"
4. Return a holistic confidence score based on the information provided

Call the tool `naics_vector_search` to search the NAICS database using vector similarity. Call this tool AT MOST twice. Based on the given context, pass ONLY ONE NAICS classifier to the tool.
Ignore the score tied to the tool. Use the tool to get the most relevant NAICS codes for the company.

Always return your final answer in JSON format with no delimiters. For example:
{
  "code": "541511",
  "industries": ["Custom Computer Programming Services"],
  "confidence": 0.8,
  "reason": "The business primarily does software consulting",
  "candidates": [{"2022 NAICS Code": "541511", "2022 NAICS Title": "Custom Computer Programming Services", "similarity_score": 0.8}]
}
""",
            retries=5,              # Increased retries
        )
        
        self.register_tools()
    
    def register_tools(self):
        @self.react_agent.tool(retries=4)
        async def naics_vector_search(ctx: RunContext[NAICSDependencies], query: str) -> Dict:
            """Search NAICS database using vector similarity"""
            try:                
                print(f"Searching NAICS database for: {query}")
                # Get hybrid search results
                matches = ctx.deps.vector_index.hybrid_search(query)
                print(f"Hybrid search results")
                if not matches:
                    raise ValueError("No matches found in search")
                
                return matches
            except Exception as e:
                raise ModelRetry(f"Search failed: {str(e)}") from e

    # def register_validators(self):
    #     @self.react_agent.result_validator
    #     def validate_naics_result(result: NAICSResponse):
    #         """Multi-field code validation"""
    #         print('result:', result)
    #         print('candidates', result.candidates)
            
    #         code_fields = ['2022 NAICS Code', 'code', 'naics_code']
    #         code_matches = any(
    #             str(c.get(field)) == result.code 
    #             for c in result.candidates 
    #             for field in code_fields
    #         )
            
    #         if not code_matches:
    #             raise ModelRetry(f"Code {result.code} not found in any candidate field")

    def test_connection(self):
        """Test OpenAI API connection"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            raise Exception(f"Failed to connect to OpenAI API: {str(e)}")
    
    def clean_search_query(self, query: str) -> str:
        """Clean and validate search query"""
        # Remove special characters and normalize whitespace
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(query))
        cleaned = ' '.join(cleaned.split())
        
        # Remove nan values
        cleaned = re.sub(r'\bnan\b', '', cleaned, flags=re.IGNORECASE)
        
        # Ensure minimum query length
        if len(cleaned.strip()) < 3:
            return None
        return cleaned.strip()

    def keyword_match_naics(self, company_name: str, naics_df: pd.DataFrame) -> Tuple[Optional[str], float, str]:
        keywords = re.findall(r'\b\w{4,}\b', company_name.lower())
        titles = naics_df['2022 NAICS Title'].fillna('').str.lower()
        
        matches = titles.apply(lambda desc: sum(1 for word in keywords if word in desc))
        
        if matches.max() > 0:
            best_match = naics_df.loc[matches.idxmax()]
            return str(best_match['2022 NAICS Code']), 0.5, "Keyword match"
        return None, 0.0, "No match found"

    def search_company_info(self, company_name: str) -> str:
        """Search for company information using Perplexity API."""
        messages = [
            {
                "role": "system",
                "content": "Provide concise business information about the specified company, including main business activities, products/services, and industry focus."
            },
            {
                "role": "user",
                "content": f"What are the main business activities and industry focus of {company_name}?"
            }
        ]
        
        try:
            response = self.perplexity_client.chat.completions.create(
                model="sonar-pro",
                messages=messages,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Perplexity API error: {str(e)}")
            return f"Error retrieving company information: {str(e)}"

    async def process_company(self, company_row: pd.Series, naics_df: pd.DataFrame) -> tuple:
        try:
            # Circuit breaker pattern
            if self.consecutive_errors > 3:
                await asyncio.sleep(2 ** self.consecutive_errors)
            
            # Add pre-validation
            if not self.validate_company_input(company_row):
                raise ClassificationError("Invalid input data")
            
            company_name = str(company_row['Company']).strip()
            city = str(company_row.get('City', '')).strip()
            website = str(company_row.get('Web Site', '')).strip()
            
            # Build search query with only valid components
            query_parts = []
            if isinstance(company_name, str) and company_name.lower() != 'nan':
                query_parts.append(company_name)
            if isinstance(city, str) and city.lower() != 'nan':
                query_parts.append(city)
            if isinstance(website, str) and website.lower() != 'nan':
                query_parts.append(website)
                
            company_info = ' '.join(query_parts) if query_parts else None
            
            if not company_info:
                return None, None, 0.0, "Invalid company information"
            
            try:
                web_content = self.search_company_info(company_name)
                response = await self.classify_with_react(company_info + ' ' + web_content)
                self.consecutive_errors = 0  # Reset on success
                
                if response.confidence < 0.2:
                    raise ClassificationError("Low confidence", response.candidates)
                    
                return (
                    response.code,
                    next((c['2022 NAICS Title'] for c in response.candidates if str(c['2022 NAICS Code']) == response.code), None),
                    response.confidence,
                    response.reason
                )
                
            except ClassificationError as e:
                self.consecutive_errors += 1
                code, confidence, reason = self.hybrid_fallback(company_name, e.candidates, naics_df)
                return code, None, confidence, reason
            except Exception as e:
                print(f"Error processing company {company_row['Company']}: {str(e)}")
                return None, None, 0.0, f"Processing error: {str(e)}"
        
        except Exception as e:
            self.log_error(e, company_row)
            return None, None, 0.0, f"Processing error: {str(e)}"

    @staticmethod
    def validate_naics_result(result: NAICSResponse, candidates: List[Dict]) -> bool:
        """Validate that selected code exists in candidates"""
        return any(str(c.get('2022 NAICS Code')) == result.code for c in candidates)

    async def classify_with_react(self, company_info: str) -> NAICSResponse:
        """Execute ReAct classification with improved limits"""
        deps = NAICSDependencies(
            vector_index=self.vector_index,
            company_info=company_info
        )
        
        result = await self.react_agent.run(
            f"Classify this business:\n{company_info}",
            deps=deps,
            usage_limits=UsageLimits(
                request_tokens_limit=10000, 
                response_tokens_limit=10000,
                total_tokens_limit=20000
            )
        )
        return result.data

    def hybrid_fallback(self, company_name: str, candidates: Optional[List[Dict]], naics_df: pd.DataFrame) -> Tuple[Optional[str], float, str]:
        """Combined fallback strategy using candidates and keywords"""
        # Try using candidates first if available
        if candidates:
            best_candidate = max(candidates, key=lambda x: x.get('similarity_score', 0))
            return (
                str(best_candidate.get('2022 NAICS Code')),
                float(best_candidate.get('similarity_score', 0.4)),
                "Fallback from candidates"
            )
        
        # Fall back to keyword matching
        return self.keyword_match_naics(company_name, naics_df)

    def validate_company_input(self, company_row: pd.Series) -> bool:
        """Validate company input data"""
        try:
            company_name = str(company_row['Company']).strip()
            return bool(company_name and company_name.lower() != 'nan')
        except:
            return False

    def log_error(self, error: Exception, company_row: pd.Series):
        """Log classification errors with context"""
        company_name = str(company_row.get('Company', 'UNKNOWN'))
        logging.error(f"Error processing {company_name}: {str(error)}")
        self.consecutive_errors += 1
