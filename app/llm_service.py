
import json
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        if not LLM_API_KEY:
            logger.warning("No API_KEY found. LLM features will be disabled.")
            self.client = None
        else:
            self.client = OpenAI(
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL
            )
            logger.info(f"LLM Service initialized with model: {LLM_MODEL}")

    def _get_completion(self, messages: List[Dict], response_format=None) -> Dict:
        """Helper to call the LLM API"""
        if not self.client:
            raise ValueError("LLM Service not available (missing API Key)")

        try:
            # Perplexity/OpenAI compatible call
            # Note: response_format={"type": "json_object"} is supported by newer OpenAI models
            # but Perplexity might handle it differently. We will ask for JSON in the prompt
            # and parse it manually to be safe across providers.
            
            completion = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.2, # Low temperature for factual analysis
            )
            
            content = completion.choices[0].message.content
            
            # Simple JSON cleanup if needed (markdown stripping)
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
                
            return json.loads(content)
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {content}")
            # Fallback: try to return a simple wrapper if JSON fails
            return {"error": "Failed to parse API response", "raw_content": content}
        except Exception as e:
            logger.error(f"LLM API Call failed: {str(e)}")
            raise e

    def understand_advisor(self, professor_name: str, abstracts: List[str]) -> Dict:
        """
        Mode 1: Summarize research trajectory and themes.
        """
        # Limit context window if needed
        context_text = "\n\n".join(abstracts[:15]) # Take top 15 abstracts to avoid token limits
        
        system_prompt = """You are an academic implementation assistant. Your goal is to analyze research papers and explain them to a prospective student.
        Output MUST be valid JSON with the following structure:
        {
            "themes": ["Theme 1", "Theme 2"],
            "trajectory": "A plain language explanation of how their work has evolved over time.",
            "summary": "A concise summary of what they work on TODAY."
        }"""
        
        user_prompt = f"""Analyze the research of Professor {professor_name}.
        Here are the abstracts of their recent papers:
        
        {context_text}
        
        Provide the analysis in the requested JSON format."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self._get_completion(messages)

    def find_research_angle(self, professor_name: str, student_interest: str, papers: List[Dict]) -> Dict:
        """
        Mode 2: Suggest research alignment.
        """
        papers_context = ""
        for i, p in enumerate(papers):
            papers_context += f"Paper {i+1}: {p['title']}\nAbstract: {p['abstract']}\n\n"
            
        system_prompt = """You are a research advisor helper. Find connections between a student's interest and a professor's work.
        Output MUST be valid JSON with the following structure:
        {
            "angles": [
                {
                    "title": "Short Title of Direction",
                    "logic": "Why this fits (referencing specific papers provided)",
                    "background_needed": "What technologies/concepts they should learn"
                }
            ]
        }
        Provide 2 distinct angles."""
        
        user_prompt = f"""Student Interest: "{student_interest}"
        
        Professor: {professor_name}
        Relevant Papers:
        {papers_context}
        
        Suggest research angles."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self._get_completion(messages)

    def generate_next_steps(self, professor_name: str, selected_angle: str, student_level: str) -> Dict:
        """
        Mode 3: Convert curiosity into action.
        """
        system_prompt = """You are a pragmatic mentor. Create a concrete action plan.
        Output MUST be valid JSON:
        {
            "checklist": ["Action 1", "Action 2", "Action 3"],
            "outreach_tips": "Advice on how to phrase an email (DO NOT write the email itself)"
        }"""
        
        user_prompt = f"""Context: A {student_level} student wants to work with Prof {professor_name} on "{selected_angle}".
        
        Give them a concrete checklist of 3-5 items to prepare before contacting the professor.
        Focus on: reading specific things, building a small demo, or checking specific prerequisites."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self._get_completion(messages)

# Singleton instance
llm_service = LLMService()
