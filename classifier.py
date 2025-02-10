from openai import OpenAI
import re
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Tuple
import pandas as pd
from utils import rate_limited, throttled
from pydantic_ai import Agent
import os
import logging

class NAICSResponse(BaseModel):
    code: str = Field(..., description="NAICS Code")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence Percentage")
    reason: str = Field(..., description="Brief Reason")

class NAICSClassifier:
    def __init__(self, openai_api_key: str, perplexity_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.perplexity_client = OpenAI(
            api_key=perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )
        self.test_connection()
        self.consecutive_errors = 0
        self.naics_analyzer = Agent(
            'openai:gpt-4o-mini',
            system_prompt="""You are an expert at NAICS classification. Follow these rules:
1. Always return valid 6-digit NAICS codes
2. If uncertain, return code 999999 with confidence <0.5
3. Format codes as strings like "541511" """,
            result_type=NAICSResponse
        )
    
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

    @throttled(base_delay=2, max_delay=30)
    @rate_limited(max_requests=10, time_window=60)
    def web_search_company(self, query):
        """Search web for company information with rate limiting"""
        try:
            # Clean and validate query
            cleaned_query = self.clean_search_query(query)
            if not cleaned_query:
                print(f"Invalid search query: {query}")
                return None
                
            # Add company-related keywords
            search_query = f"{cleaned_query} company business"
            
            results = list(self.ddg.text(
                keywords=search_query,
                region="wt-wt",
                safesearch=False,
                max_results=10
            ))
            
            if not results:
                print(f"No results found for: {search_query}")
                return None
                
            return results
            
        except Exception as e:
            print(f"Error searching {query}: {str(e)}")
            return None

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
                model="sonar",
                messages=messages,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Perplexity API error: {str(e)}")
            return f"Error retrieving company information: {str(e)}"

    async def process_company(self, company_row: pd.Series, naics_df: pd.DataFrame) -> tuple:
        current_delay = None
        try:
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
            except Exception as e:
                if "Rate limit" in str(e):
                    return None, None, 0.0, f"Rate limited: {str(e)}"
                return None, None, 0.0, f"Search failed: {str(e)}"
            
            if not isinstance(web_content, (str, list)) or not web_content:
                return None, None, 0.0, "No web content found"
                
            # Convert list of search results to string if needed
            if isinstance(web_content, list):
                web_content = ' '.join([str(r) for r in web_content[:3]])
            
            if web_content:
                try:
                    analysis_result = (await self.naics_analyzer.run(
                        f"Company: {company_name}\nContext: {web_content}"
                    )).data

                    # Ensure code format validation
                    if not re.match(r"^\d{6}$", analysis_result.code):
                        raise ValidationError(f"Invalid NAICS code format: {analysis_result.code}")

                    code = analysis_result.code
                    confidence = analysis_result.confidence
                    reason = analysis_result.reason
                    source = 'Web Analysis'

                except ValidationError as e:
                    logging.warning(f"Validation failed for {company_name}: {e}")
                    code = "999999"
                    confidence = 0.0
                    reason = f"Validation error: {e}"
                    source = 'Web Analysis'
                except Exception as e:
                    logging.error(f"API error: {str(e)}")
                    code = "999999"
                    confidence = 0.0
                    reason = f"API error: {str(e)}"
                    source = 'Web Analysis'
            else:
                code, confidence, reason = self.keyword_match_naics(company_name, naics_df)
                source = 'Keyword Match'
            
            description = None
            if code and code in naics_df['2022 NAICS Code'].astype(str).values:
                description = naics_df[naics_df['2022 NAICS Code'].astype(str) == code]['2022 NAICS Title'].iloc[0]
            
            return code, description, confidence, f"{source}: {reason}"
        
        except Exception as e:
            print(f"Error processing company {company_row['Company']}: {str(e)}")
            return None, None, 0.0, f"Processing error: {str(e)}"
