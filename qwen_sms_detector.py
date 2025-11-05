"""
Qwen-based SMS Phishing Detector

This module implements a comprehensive SMS phishing detection system similar to
SMSPhishingDetector in main.py, but uses the locally run Qwen2-VL-7B-Instruct
model instead of OpenAI's API. It processes CSV files of SMS messages and uses
the same APIs (Jina, Google Cloud) for data gathering.

Usage:
    from config import jina_api_key, google_cloud_API_key, search_engine_ID
    detector = QwenSMSPhishingDetector(
        jina_api_key=jina_api_key,
        google_cloud_API_key=google_cloud_API_key,
        search_engine_id=search_engine_ID
    )
    detector.detect_sms_phishing("Your SMS message here", "output")
"""

import csv
import json
import os
import requests
import base64
import subprocess
import re
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import jina_api_key, google_cloud_API_key, search_engine_ID, http_request_header

# Optional whois import
try:
    import whois
except ImportError:
    whois = None
    print("Warning: python-whois not installed. Domain info will be unavailable.")


class QwenSMSPhishingDetector:
    """
    A comprehensive SMS phishing detection system that analyzes SMS messages
    using the locally run Qwen2-VL-7B-Instruct model instead of OpenAI's API.
    Uses the same APIs (Jina, Google Cloud) for data gathering.
    """
    
    def __init__(
        self,
        jina_api_key: str,
        google_cloud_API_key: str,
        search_engine_id: str,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "auto",
        trust_remote_code: bool = True
    ):
        """
        Initialize the Qwen-based SMS Phishing Detector.
        
        Args:
            jina_api_key (str): Jina API key for web content extraction
            google_cloud_API_key (str): Google Cloud API key for search
            search_engine_id (str): Google Custom Search Engine ID
            model_name: HuggingFace model name for Qwen2-VL
            device: Device to use ("auto", "cpu", "cuda")
            trust_remote_code: Whether to trust remote code when loading model
        """
        self.jina_api_key = jina_api_key
        self.google_cloud_API_key = google_cloud_API_key
        self.search_engine_id = search_engine_id
        self.model_name = model_name
        # Set device - use cuda if available and requested, otherwise cpu
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.trust_remote_code = trust_remote_code
        
        print(f"Loading Qwen model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            
            # For Qwen2-VL, try to load the specific model class with memory optimization
            try:
                from transformers import Qwen2VLForConditionalGeneration
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True,
                    max_memory={0: "10GB", 1: "10GB", 2: "10GB", 3: "10GB"} if self.device == "cuda" else None
                )
            except (ImportError, ValueError, Exception) as e:
                print(f"Error loading Qwen2VLForConditionalGeneration: {e}")
                # For Qwen2-VL, we need to use the specific model class, not AutoModel
                # Let's try a simpler approach without device_map for now
                try:
                    from transformers import Qwen2VLForConditionalGeneration
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name,
                        trust_remote_code=trust_remote_code,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    )
                    if self.device == "cuda":
                        self.model = self.model.cuda()
                except Exception as e2:
                    print(f"Error with simple Qwen2VL loading: {e2}")
                    # This model requires Qwen2VLForConditionalGeneration, no fallback
                    raise ValueError(f"Qwen2-VL model requires Qwen2VLForConditionalGeneration class. Error: {e2}")
            
            # Don't move model if it's already loaded with device_map
            if not hasattr(self.model, 'hf_device_map'):
                self.model = self.model.to(self.device)
            self.model.eval()
            
            print("Qwen model loaded successfully")
        except Exception as e:
            print(f"Error loading Qwen model: {e}")
            print("Make sure the model is downloaded and accessible")
            raise
        
    def detect_sms_phishing(
        self, 
        sms_message: str, 
        output_dir: str = "output",
        enable_redirect_chain: bool = True,
        enable_brand_search: bool = True,
        enable_screenshot: bool = True,
        enable_html_content: bool = True,
        enable_domain_info: bool = True
    ) -> bool:
        """
        Main function to detect if an SMS message is phishing.
        
        Args:
            sms_message (str): The SMS message to analyze
            output_dir (str): Directory to save analysis results
            enable_redirect_chain (bool): Whether to analyze URL redirect chains
            enable_brand_search (bool): Whether to search for brand domains
            enable_screenshot (bool): Whether to take website screenshots
            enable_html_content (bool): Whether to analyze HTML content
            enable_domain_info (bool): Whether to get domain information
            
        Returns:
            bool: True if the SMS is detected as phishing/spam, False if legitimate
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Extract URLs and brands from SMS using Qwen
        initial_analysis = self._extract_urls_and_brands(sms_message)
        
        # Step 2: Analyze URLs if present
        if initial_analysis['is_URL'] and initial_analysis['URLs'] != "non":
            url_analysis = self._analyze_urls(
                initial_analysis['URLs'],
                output_dir,
                enable_redirect_chain,
                enable_brand_search,
                enable_screenshot,
                enable_html_content,
                enable_domain_info
            )
            initial_analysis['URLs'] = url_analysis
            
            # Brand search if enabled and brands detected
            if enable_brand_search and initial_analysis['is_brand']:
                brand_analysis = self._search_brand_domains(initial_analysis['brands'])
                for url_idx in url_analysis:
                    initial_analysis['URLs'][url_idx]['brand_search'] = brand_analysis
        
        # Step 3: Generate detection prompt and analyze with Qwen
        detection_prompt = self._build_detection_prompt(sms_message, initial_analysis)
        detection_result = self._perform_final_detection(detection_prompt)
        
        # Step 4: Generate user-friendly output with Qwen
        user_friendly_output = self._generate_user_friendly_output(sms_message, detection_result)
        
        # Step 5: Save complete analysis
        complete_analysis = {
            **initial_analysis,
            'SMS': sms_message,
            'detect_result': detection_result,
            'user_friendly_output': user_friendly_output,
            'detection_prompt': detection_prompt,
            'model_used': 'Qwen2-VL-7B-Instruct'
        }
        
        self._save_analysis_results(complete_analysis, output_dir)
        
        return detection_result.get('category', True)
    
    def _extract_urls_and_brands(self, sms_message: str) -> Dict:
        """Extract URLs and brand names from SMS message using Qwen."""
        prompt = self._get_url_extraction_prompt() + "\n" + sms_message
        
        try:
            response = self._query_qwen(prompt, max_new_tokens=256)
            # Try to extract JSON from response
            content = self._clean_json_response(response)
            result = json.loads(content)
            return result
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error extracting URLs and brands: {e}")
            # Fallback: try to extract URLs manually
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', sms_message)
            return {
                'is_URL': len(urls) > 0,
                'URLs': urls if urls else "non",
                'is_brand': False,
                'brands': "non"
            }
    
    def _analyze_urls(
        self, 
        urls: List[str], 
        output_dir: str,
        enable_redirect_chain: bool,
        enable_brand_search: bool,
        enable_screenshot: bool,
        enable_html_content: bool,
        enable_domain_info: bool
    ) -> Dict:
        """Analyze each URL in the SMS message."""
        url_analysis = {}
        
        for idx, url in enumerate(urls):
            url_analysis[idx] = {'URL': url}
            
            # Normalize URL
            normalized_url = self._normalize_url(url)
            final_url = self._expand_url(normalized_url)
            
            url_analysis[idx]['final_URL'] = final_url or normalized_url
            
            # Redirect chain analysis
            if enable_redirect_chain:
                redirect_chain = self._get_redirect_chain(normalized_url)
                url_analysis[idx]['redirect_chain'] = redirect_chain
            
            # HTML content analysis (using Jina API like original)
            if enable_html_content:
                html_content, html_summary = self._analyze_html_content(final_url or normalized_url)
                url_analysis[idx]['URL_content'] = html_content
                url_analysis[idx]['html_summary'] = html_summary
            
            # Domain information
            if enable_domain_info:
                domain_info = self._get_domain_info(normalized_url)
                url_analysis[idx]['domain_info'] = domain_info
            
            # Screenshot analysis
            if enable_screenshot:
                screenshot_path, image_content = self._analyze_screenshot(
                    final_url or normalized_url, 
                    output_dir, 
                    idx
                )
                url_analysis[idx]['screenshot_path'] = screenshot_path
                url_analysis[idx]['Image_content'] = image_content
        
        return url_analysis
    
    def _normalize_url(self, url: str) -> str:
        """Add http:// prefix if missing."""
        if not (url.startswith("http://") or url.startswith("https://")):
            return "http://" + url
        return url
    
    def _check_url_validity(self, url: str) -> Tuple[bool, Optional[int]]:
        """Check if URL is valid and accessible."""
        try:
            response = requests.head(url, allow_redirects=True, headers=http_request_header)

            # If the status code is in the range of 200 to 399, the URL is valid
            if response.status_code in range(200, 400):
                return True, response.status_code
            else:
                return False, response.status_code
        except requests.exceptions.RequestException as e:
            print(f"Error occurred: {e}")
            return False, None
    
    def _expand_url(self, url: str) -> Optional[str]:
        """Expand shortened URLs to their final destination."""
        try:
            response = requests.head(url, allow_redirects=True, headers=http_request_header, timeout=10)
            return response.url
        except requests.RequestException:
            return None
    
    def _get_redirect_chain(self, url: str) -> List[Tuple[str, int]]:
        """Get the complete redirect chain for a URL."""
        try:
            response = requests.head(url, allow_redirects=True, headers=http_request_header)

            response_chain = []
            response_status = []

            if response.history:
                for resp in response.history:
                    response_chain.append(resp.url)
                    response_status.append(resp.status_code)

            # Add the final response URL and status
            response_chain.append(response.url)
            response_status.append(response.status_code)

            return list(zip(response_chain, response_status))
        except requests.RequestException:
            return "non"
    
    def _analyze_html_content(self, url: str) -> Tuple[str, str]:
        """
        Analyze HTML content of the URL using Jina API (same as original),
        then summarize with Qwen.
        """
        try:
            jina_url = f'https://r.jina.ai/{url}'
            headers = {"Authorization": f"Bearer {self.jina_api_key}"}
            response = requests.get(jina_url, headers=headers)
            
            # Limit content length
            content = response.text[:10000] if len(response.text) > 10000 else response.text
            
            # Summarize content using Qwen instead of GPT
            summary = self._summarize_html_content(content)
            return content, summary
            
        except requests.RequestException:
            content = "There is no information known about the URL. The URL might be invalid or expired."
            return content, content
    
    def _summarize_html_content(self, content: str) -> str:
        """Summarize HTML content using Qwen model."""
        prompt = """Please summarize the following website content in English and determine whether the website has a block wall or not.
Your output should be in JSON format:
{
  "summary": "the summary of the content in English. Within 500 words. Some website might have a robot-human verification page. If the website has no information available, mention that the content might be hidden behind a verification wall. Both phishing and legitimate websites can have a robot-human verification page. It doesn't necessarily indicate malicious intent."
}

Website content:
""" + content[:5000]  # Limit content length
        
        try:
            response = self._query_qwen(prompt, max_new_tokens=512)
            
            # Try to extract JSON from response
            if 'summary' in response.lower():
                json_match = re.search(r'\{[^}]*"summary"[^}]*\}', response, re.DOTALL)
                if json_match:
                    try:
                        summary_data = json.loads(json_match.group(0))
                        return summary_data.get('summary', response[:500])
                    except json.JSONDecodeError:
                        pass
                
                # Fallback: extract text after "summary"
                summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', response)
                if summary_match:
                    return summary_match.group(1)
            
            return response[:500]  # Return first 500 chars as summary
        except Exception as e:
            print(f"Error summarizing HTML content with Qwen: {e}")
            return "Could not analyze content"
    
    def _get_domain_info(self, url: str) -> str:
        """Get domain registration information."""
        if whois is None:
            return "non"
        try:
            domain = url.split("//")[-1].split("/")[0]
            domain_info = whois.whois(domain)
            return str(domain_info)
        except Exception:
            return "non"
    
    def _analyze_screenshot(self, url: str, output_dir: str, idx: int) -> Tuple[str, str]:
        """Take and analyze screenshot of the webpage using Qwen Vision."""
        try:
            screenshot_path = os.path.join(output_dir, f"screenshot_{idx}.png")
            
            # Take screenshot if it doesn't exist
            if not os.path.exists(screenshot_path):
                self._take_screenshot(url, screenshot_path)
            
            # Analyze screenshot with Qwen Vision
            image_content = self._analyze_screenshot_with_qwen(screenshot_path)
            return screenshot_path, image_content
            
        except Exception as e:
            print(f"Screenshot error: {e}")
            return "non", "non"
    
    def _take_screenshot(self, url: str, screenshot_path: str):
        """Take screenshot using Node.js crawler (same as original)."""
        try:
            # Set up environment with conda environment's bin directory in PATH
            env = os.environ.copy()
            conda_env_path = "/home/myid/zl26271/.conda/envs/SmishX/bin"
            if conda_env_path not in env.get('PATH', ''):
                env['PATH'] = f"{conda_env_path}:{env.get('PATH', '')}"
            
            subprocess.run(
                ['node', 'crawler_proj/crawler.js', url, screenshot_path],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            print(f"Screenshot saved to {screenshot_path}")
        except subprocess.SubprocessError as e:
            print(f"Screenshot capture failed: {e}")
            raise
    
    def _analyze_screenshot_with_qwen(self, image_path: str) -> str:
        """Analyze screenshot using Qwen Vision capabilities."""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            prompt = """You are a website screenshot analysis assistant. Analyze this website screenshot and provide a detailed description of the content, determining the purpose of the page.
- If the screenshot shows a news site, summarize the main news topics or articles.
- Identify any logos, brands, or key visual elements.
- The URL might be redirected to a robot-human verification page. If the screenshot is a blank page, mention that the content might be hidden behind a verification wall.
Your response should be in English and plain text, without any markdown or HTML formatting. Your response should be in 15 sentences or less."""
            
            try:
                # Use Qwen's vision capabilities
                response = self._query_qwen_vision(image, prompt, max_new_tokens=300)
                return response
            except Exception as e:
                print(f"Vision query failed, using text-only fallback: {e}")
                # Fallback to text-only description
                return self._query_qwen(f"{prompt}\n\nNote: I cannot see the image, but please provide general guidance based on the description.", max_new_tokens=300)
        
        except Exception as e:
            print(f"Error analyzing screenshot with Qwen: {e}")
            return "non"
    
    def _search_brand_domains(self, brands: List[str]) -> Dict:
        """Search for official domains of mentioned brands using Google API (same as original)."""
        brand_search = {}
        
        for idx, brand in enumerate(brands):
            brand_search[idx] = {
                'brand_name': brand,
                'brand_domain': self._google_search_brand(brand_name=brand, google_cloud_API=self.google_cloud_API_key, search_engine_id=self.search_engine_id)
            }
        
        return brand_search

    def _google_search_brand(self, google_cloud_API: str, search_engine_id: str, brand_name: str) -> List[str]:
        """Search Google for brand's official domains (same as original)."""
        try:
            url = f"https://www.googleapis.com/customsearch/v1"
            params = {
                'key': google_cloud_API,
                'cx': search_engine_id,
                'q': brand_name,
                'num': 5,  # Get top 5 results
            }
            response = requests.get(url, params=params)
            response = response.json()
            return [item['link'] for item in response.get('items', [])]
        except Exception as e:
            print(f"Google search failed: {e}")
            return []

    def _build_detection_prompt(self, sms_message: str, analysis: Dict) -> str:
        """Build the comprehensive prompt for final detection (same structure as original)."""
        prompt = self._get_detection_prompt_template()
        prompt += f"\n- SMS to be analyzed: {sms_message}\n"
        
        if analysis.get('is_URL') and analysis.get('URLs') != "non":
            urls = analysis.get('URLs', {})
            if len(urls) > 1:
                prompt += f"- There are {len(urls)} URLs in the SMS.\n"
            
            for url_idx, url_data in urls.items():
                url = url_data.get('URL', '')
                if len(urls) > 1:
                    prompt += f"- URL {url_idx}: {url}\n"
                else:
                    prompt += f"- URL: {url}\n"
                
                # Add analysis data if available
                if url_data.get('redirect_chain') not in [None, "non"]:
                    prompt += f"- Redirect Chain of {url}: {url_data['redirect_chain']}\n"
                
                if url_data.get('html_summary') not in [None, "non"]:
                    prompt += f"- Html Content Summary of {url}: {url_data['html_summary']}\n"
                
                if url_data.get('domain_info') not in [None, "non"]:
                    prompt += f"- Domain Information of {url}: {url_data['domain_info']}\n"
                
                if url_data.get('Image_content') not in [None, "non"]:
                    prompt += f"- Screenshot Description {url}: {url_data['Image_content']}\n"
                
                if url_data.get('brand_search') not in [None, "non"] and analysis.get('is_brand'):
                    brands = analysis.get('brands', [])
                    if len(brands) > 1:
                        prompt += f"- There are {len(brands)} brands referred in the SMS.\n"
                    
                    for brand_idx, brand in enumerate(brands):
                        prompt += f"- Brand {brand_idx}: {brand}\n"
                        brand_domains = url_data.get('brand_search', {}).get(brand_idx, {}).get('brand_domain', [])
                        prompt += f"- The top five results from a Google search of the brand name: {brand_domains}\n"
        else:
            prompt += "- No URL in the SMS.\n"
        
        return prompt
    
    def _perform_final_detection(self, prompt: str) -> Dict:
        """Perform final phishing detection using Qwen model."""
        try:
            response = self._query_qwen(prompt, max_new_tokens=512)
            
            # Try to extract JSON from response
            detection_result = self._extract_json_from_response(response)
            
            # Ensure required fields
            if 'category' not in detection_result:
                # Try to infer category from response text
                response_lower = response.lower()
                if 'legitimate' in response_lower or 'false' in response_lower:
                    detection_result['category'] = False
                else:
                    detection_result['category'] = True
            
            # Set defaults for missing fields
            detection_result.setdefault('brief_reason', 'Analysis completed')
            detection_result.setdefault('rationales', response[:500] if 'rationales' not in detection_result else detection_result['rationales'])
            detection_result.setdefault('advice', 'Please exercise caution')
            detection_result.setdefault('URL', 'non')
            detection_result.setdefault('brand_impersonated', '')
            
            return detection_result
            
        except Exception as e:
            print(f"Detection analysis failed: {e}")
            return {
                "category": True,
                "brief_reason": "Analysis failed",
                "advice": "Exercise caution",
                "rationales": f"Error during analysis: {str(e)}",
                "URL": "non",
                "brand_impersonated": ""
            }
    
    def _generate_user_friendly_output(self, sms_message: str, detection_result: Dict) -> str:
        """Generate user-friendly analysis output using Qwen."""
        prompt = self._get_user_friendly_prompt() + f"\nThe SMS message: {sms_message}\nThe analysis result: {detection_result}"
        
        try:
            response = self._query_qwen(prompt, max_new_tokens=200)
            return self._clean_json_response(response)
        except Exception:
            return "Unable to generate user-friendly analysis."
    
    def _save_analysis_results(self, analysis: Dict, output_dir: str):
        """Save complete analysis results to JSON file."""
        # Clean up data for JSON serialization
        cleaned_analysis = self._prepare_for_json_serialization(analysis)
        
        output_file = os.path.join(output_dir, "analysis_output.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_analysis, f, indent=2, ensure_ascii=False)
    
    def _prepare_for_json_serialization(self, data):
        """Prepare data for JSON serialization by handling special types."""
        from datetime import datetime
        from requests.structures import CaseInsensitiveDict
        
        if isinstance(data, CaseInsensitiveDict):
            return dict(data)
        elif isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, set):
            return list(data)
        elif isinstance(data, dict):
            return {k: self._prepare_for_json_serialization(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json_serialization(item) for item in data]
        else:
            return data
    
    def _clean_json_response(self, content: str) -> str:
        """Clean Qwen response to extract valid JSON or text."""
        content = content.replace("\n", "")
        content = content.replace("```json", "")
        content = content.replace("```", "")
        return content.strip()
    
    def _get_url_extraction_prompt(self) -> str:
        """Get prompt for URL and brand extraction."""
        return """Extract any URLs, and brand names from the following SMS message.
Your output should be in JSON format and should not have any other output: 
- is_URL: true or false
- URLs: if no URL in SMS, answer "non". If there are URLs, the response should be a list, each element is a URL extracted from the SMS.
- is_brand: true or false
- brands: if no brand name in SMS, answer "non". If there are brand names, the response should be a list, each element is a brand name extracted from the SMS. You can extract the brand name from the SMS content and the URL."""
    
    def _get_detection_prompt_template(self) -> str:
        """Get the main detection prompt template (same as original)."""
        return """I want you to act as a spam detector to determine whether a given SMS is phishing, spam, or legitimate. Your analysis should be thorough and evidence-based. Analyze the SMS by following these steps:
1. If the SMS is promoting any of the following categories: Online gambling, bets, spins, adult content, digital currency, lottery, it is either spam or phishing.
2. The SMS is legitimate if it is from known organizations, such as appointment reminders, OTP (One-Time Password) verification, delivery notifications, account updates, tracking information, or other expected messages.
3. The SMS is considered legitimate if it involves a conversation between friends, family members, or colleagues.
4. Promotions and advertisements are considered spam. The SMS is spam if it is promotion from legitimate companies and is not impersonating any brand, but it is advertisements, app download promotions, sales promotions, donation requests, event promotions, online loan services, or other irrelevant information.
5. The SMS is phishing if it is fraudulent and attempts to deceive recipients into providing sensitive information or clicking malicious links. Phishing SMS may exhibit the following characteristics:
Promotions or Rewards: Some phishing SMS offer fake prizes, rewards, or other incentives to lure recipients into clicking links or providing personal information.
Urgent or Alarming Language: Phishing messages often create a sense of urgency or fear, such as threats of account suspension, missed payments, or urgent security alerts.
Suspicious Links: Phishing messages may contain links to fake websites designed to steal personal information.
Requests for Personal Information: Phishing SMS may ask for sensitive information like passwords, credit card numbers, social security numbers, or other personal details.
Grammatical and Spelling Errors: Many phishing messages contain grammatical mistakes or unusual wording, which can be a red flag for recipients.
Expired Domain: Phishing websites often use domains that expire quickly or are already listed for sale.
Inconsistency: The URL may be irrelevant to the message content.
6. Please be aware that: It is common to see shortened URLs in SMS. You can get the expanded URL from the provided redirection chain. Both phishing and legitimate URLs can be shortened. And both phishing and legitimate websites may use a robot-human verification page (CAPTCHA-like mechanism) before granting access the content.
7. I will provide you with some external information if there is a URL in the SMS. The information includes:
- Redirect Chain: The URL may redirect through multiple intermediate links before reaching the final destination; if any of them is flagged as phishing, the original URL becomes suspicious.
- Brand Search Information: The top five results from a Google search of the brand name. You can compare if the URL's domain matches the results from Google.
- Screenshot Description: A description of the website's screenshot, highlighting any notable visual elements.
- HTML Content Summary: The title of HTML, and the summary of its content.
- Domain Information: The domain registration details, including registrar, creation date, and DNS records, which are analyzed to verify the domain's legitimacy.
8. Please give your rationales before making a decision. And your output should be in JSON format and should not have any other output:
- brand_impersonated: brand name associated with the SMS, if applicable.
- URL: any URL appears in SMS, if no URL, answer "non".
- rationales: detailed rationales for the determination, up to 500 words. Directly give sentences, do not categorize the rationales. Only tell the reasons why the SMS is legitimate or not, do not include the reasons why the SMS is spam or phishing.
- brief_reason: brief reason for the determination.
- category: True or False. If the SMS is legitimate, output False. Else, output True.
- advice: If the SMS is phishing, output potential risk and your advice for the recipients, such as "Do not respond to this message or access the link."

Below is the information of the SMS:"""
    
    def _get_user_friendly_prompt(self) -> str:
        """Get prompt for generating user-friendly output."""
        return """Based on the detailed analysis, I want you to create a simple and easy-to-understand response to tell the user whether the text message is a phishing attempt or a legitimate message. Use plain language and avoid technical terms like URL or HTTP headers. Explain your conclusion in 3 sentences, focusing on whether the message seems suspicious or safe. Provide a simple reason to support your conclusion, including clear evidence such as a suspicious website link or an urgent tone in the message. The response should be reassuring and concise, easy for anyone to understand."""
    
    def _query_qwen(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Query Qwen model with text-only prompt."""
        try:
            # Prepare messages in Qwen's chat format
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id
                )
            
            # Decode response
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids
                in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()
            
        except Exception as e:
            print(f"Error querying Qwen: {e}")
            raise
    
    def _query_qwen_vision(self, image: Image.Image, prompt: str, max_new_tokens: int = 300) -> str:
        """Query Qwen model with image and text prompt."""
        try:
            # Prepare messages with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template (Qwen2-VL supports this format)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Prepare inputs (may need special handling for vision models)
            try:
                # Try Qwen2-VL specific processing
                inputs = self.tokenizer(text=[text], images=[image], return_tensors="pt")
            except:
                # Fallback to text-only
                inputs = self.tokenizer([text], return_tensors="pt")
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id
                )
            
            # Decode
            if isinstance(generated_ids, list):
                generated_ids = generated_ids[0]
            
            input_ids_len = inputs['input_ids'].shape[1]
            generated_ids = generated_ids[input_ids_len:]
            
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            print(f"Error querying Qwen Vision: {e}")
            # Fallback to text-only
            return self._query_qwen(prompt, max_new_tokens=max_new_tokens)
    
    def _extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON from Qwen's response text."""
        # Remove markdown code blocks if present
        response = response.replace("```json", "").replace("```", "")
        
        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try to parse entire response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Fallback: create structure from text
        result = {}
        response_lower = response.lower()
        
        # Extract category
        if 'category' in response_lower or '"category"' in response:
            cat_match = re.search(r'"category"\s*:\s*(true|false)', response, re.IGNORECASE)
            if cat_match:
                result['category'] = cat_match.group(1).lower() == 'true'
            elif 'false' in response_lower and 'legitimate' in response_lower:
                result['category'] = False
            else:
                result['category'] = True
        else:
            result['category'] = True
        
        # Extract other fields
        for field in ['brief_reason', 'rationales', 'advice', 'URL', 'brand_impersonated']:
            field_match = re.search(rf'"{field}"\s*:\s*"([^"]+)"', response)
            if field_match:
                result[field] = field_match.group(1)
            else:
                if field == 'rationales':
                    result[field] = response[:500]
                elif field == 'brief_reason':
                    result[field] = 'Analysis completed'
                elif field == 'advice':
                    result[field] = 'Exercise caution'
                else:
                    result[field] = ''
        
        return result


def evaluate_detector_on_csv(csv_path: str, x: int) -> float:
    """
    Evaluate the QwenSMSPhishingDetector on the first x rows of a CSV file.
    
    The CSV is expected to have the SMS text in the first column and the label
    in the second column. If the label is "legitimate", the ground truth is False
    (not phishing). Any other label is treated as True (phishing).
    
    Args:
        csv_path (str): Absolute path to the CSV file.
        x (int): Number of rows from the top (after header) to evaluate.
        
    Returns:
        float: Accuracy percentage as (correct / (correct + incorrect)) * 100.
    """
    detector = QwenSMSPhishingDetector(
        jina_api_key=jina_api_key,
        google_cloud_API_key=google_cloud_API_key,
        search_engine_id=search_engine_ID,
    )
    
    total_evaluated = 0
    num_correct = 0
    num_incorrect = 0
    
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # Skip header if present
        header = next(reader, None)
        # Heuristically detect if first row is a header
        if header and len(header) >= 2 and header[0].strip().lower() == "sms" and header[1].strip().lower() == "label":
            pass
        else:
            # If not a header, process it as data
            if header is not None:
                row = header
                if len(row) >= 2 and total_evaluated < x:
                    sms_text = row[0]
                    label = row[1]
                    ground_truth_is_phishing = (label.strip().lower() != "legitimate")
                    predicted_is_phishing = bool(detector.detect_sms_phishing(sms_text, output_dir=f"qwen_output_{total_evaluated}"))
                    if predicted_is_phishing == ground_truth_is_phishing:
                        num_correct += 1
                    else:
                        num_incorrect += 1
                    total_evaluated += 1
                    
                    # Print progress every 5 evaluations
                    if total_evaluated % 5 == 0 or total_evaluated == 1:
                        current_accuracy = (num_correct / total_evaluated) * 100 if total_evaluated > 0 else 0
                        print(f"Progress: {total_evaluated}/{x} rows evaluated | Current accuracy: {current_accuracy:.2f}%")
        
        for idx, row in enumerate(reader):
            if total_evaluated >= x:
                break
            if len(row) < 2:
                continue
            # Skip every other row (read only even-indexed rows)
            if idx % 2 != 0:
                continue
            sms_text = row[0]
            label = row[1]
            # ground truth is phishing if the label is EITHER phishing or spam
            ground_truth_is_phishing = (label.strip().lower() != "legitimate")
            predicted_is_phishing = bool(detector.detect_sms_phishing(sms_text, output_dir=f"qwen_output_{total_evaluated}"))
            if predicted_is_phishing == ground_truth_is_phishing:
                num_correct += 1
            else:
                num_incorrect += 1
            total_evaluated += 1
            
            # Print progress every 5 evaluations
            if total_evaluated % 5 == 0 or total_evaluated == 1:
                current_accuracy = (num_correct / total_evaluated) * 100 if total_evaluated > 0 else 0
                print(f"Progress: {total_evaluated}/{x} rows evaluated | Current accuracy: {current_accuracy:.1f}%")
    
    if total_evaluated == 0:
        return 0.0
    return (num_correct / total_evaluated) * 100.0


def test_detector_on_file(file_path: str, rows: int):
    """Test the Qwen detector on a CSV file."""
    detector = QwenSMSPhishingDetector(
        jina_api_key=jina_api_key,
        google_cloud_API_key=google_cloud_API_key,
        search_engine_id=search_engine_ID,
    )
    result = "Accuracy on rows: " + str(evaluate_detector_on_csv(file_path, rows))
    return result


def test_detector_on_text(text: str) -> str:
    """Test the Qwen detector on a single SMS text."""
    detector = QwenSMSPhishingDetector(
        jina_api_key=jina_api_key,
        google_cloud_API_key=google_cloud_API_key,
        search_engine_id=search_engine_ID,
    )
    result = detector.detect_sms_phishing(text, "qwen_analysis_output")
    return result


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Qwen-based SMS Phishing Detector')
    parser.add_argument(
        '--csv_path',
        type=str,
        help='Path to CSV file with SMS messages (first column: SMS, second column: label)'
    )
    parser.add_argument(
        '--num_rows',
        type=int,
        default=10,
        help='Number of rows to evaluate'
    )
    parser.add_argument(
        '--sms_text',
        type=str,
        help='Single SMS message to analyze'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen/Qwen2-VL-7B-Instruct',
        help='Qwen model name or path'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    if args.csv_path:
        print(f"Evaluating Qwen detector on CSV file: {args.csv_path}")
        print(f"Testing on {args.num_rows} rows...")
        result = test_detector_on_file(args.csv_path, args.num_rows)
        print(result)
    elif args.sms_text:
        print(f"Analyzing SMS with Qwen detector: {args.sms_text[:100]}...")
        result = test_detector_on_text(args.sms_text)
        print(f"Detection result: {result}")
    else:
        print("Please provide either --csv_path or --sms_text argument")
        print("\nExample usage:")
        print("  python qwen_sms_detector.py --sms_text 'Your SMS message here'")
        print("  python qwen_sms_detector.py --csv_path data/dataset.csv --num_rows 10")