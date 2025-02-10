from openai import OpenAI
import re
from pydantic import BaseModel, Field
from typing import Optional, Tuple
import pandas as pd
from duckduckgo_search import DDGS

class NAICSResponse(BaseModel):
    code: str = Field(..., alias="NAICS Code")
    confidence: float = Field(..., ge=0.0, le=1.0, alias="Confidence Percentage")
    reason: str = Field(..., alias="Brief Reason")

class NAICSClassifier:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def web_search_company(query):
        """Search web for company information"""
        try:
            # Get top 3 Google results
            results = list(DDGS().text(
                    keywords=query, 
                    region="wt-wt", 
                    safesearch=False, 
                    max_results=10
            ))
            return results
        except Exception as e:
            print(f"Error searching {query}: {str(e)}")
        return None

    def analyze_with_gpt(self, context: str, company_name: str) -> Tuple[Optional[str], float, str]:
        try:
            prompt = f"""Analyze this company information and return NAICS classification:
            Company: {company_name}
            Context: {context}"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at NAICS classification."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = NAICSResponse.parse_raw(response.choices[0].message.content)
            return result.code, result.confidence, result.reason
        except Exception as e:
            print(f"GPT analysis error: {str(e)}")
            return None, 0.0, str(e)

    def keyword_match_naics(self, company_name: str, naics_df: pd.DataFrame) -> Tuple[Optional[str], float, str]:
        keywords = re.findall(r'\b\w{4,}\b', company_name.lower())
        titles = naics_df['2022 NAICS Title'].fillna('').str.lower()
        
        matches = titles.apply(lambda desc: sum(1 for word in keywords if word in desc))
        
        if matches.max() > 0:
            best_match = naics_df.loc[matches.idxmax()]
            return str(best_match['2022 NAICS Code']), 0.5, "Keyword match"
        return None, 0.0, "No match found"

    def process_company(self, company_row: pd.Series, naics_df: pd.DataFrame) -> tuple:
        company_name = company_row['Company']
        company_info = f"{company_name} {company_row.get('City', '')} {company_row.get('Web Site', '')}"
        
        web_content = self.web_search_company(company_info)
        
        if web_content:
            code, confidence, reason = self.analyze_with_gpt(web_content, company_name)
            source = 'Web Analysis'
        else:
            code, confidence, reason = self.keyword_match_naics(company_name, naics_df)
            source = 'Keyword Match'
        
        description = None
        if code and code in naics_df['2022 NAICS Code'].astype(str).values:
            description = naics_df[naics_df['2022 NAICS Code'].astype(str) == code]['2022 NAICS Title'].iloc[0]
        
        return code, description, confidence, f"{source}: {reason}"
