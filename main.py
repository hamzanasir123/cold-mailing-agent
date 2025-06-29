import random
import time
import uuid
import nest_asyncio
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner
from openai import AsyncOpenAI
from dotenv import load_dotenv
from datetime import datetime
from supabase import create_client
import requests
import schedule
import os
import html
import json
import re
import pandas as pd

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
brevo_api_key = os.getenv("BREVO_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize external client and model
external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Initialize agent
agent = Agent(
    name="Cold_Mailing_Agent",
    model=model,
    instructions="You are a cold mailing agent. You will generate a cold mail for a given company. Please make sure the mail is professional and the company is in the tech industry."
)

# Profile
profile = """
We're a development team offering services in:
- JAMstack, MERNstack, Frontend, and Backend Development
- AI Agents
- Voice & Customer Support AI
- AI Automated Workflows
- Smart Contracts, Blockchain
- Shopify & WordPress
- Debugging, Migrations, API Development
"""

# Functions
def extract_json_from_output(output: str):
    cleaned = re.sub(r"```(?:json)?", "", output).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            match = re.search(r"\[\s*{.*?}\s*\]", cleaned, re.DOTALL)
            if match:
                json_array = match.group()
                return json.loads(json_array)
        except Exception as inner:
            print(f"Failed to parse JSON: {inner}")
        with open("json_debug_output.txt", "w", encoding="utf-8") as f:
            f.write(output)
        raise json.JSONDecodeError("Unable to decode valid JSON", cleaned, 0)

def is_duplicate(email):
    try:
        response = supabase.table("cold_mailing_agent").select("email").eq("email", email).execute()
        return len(response.data) > 0
    except Exception as e:
        print(f"Error checking duplicate email: {str(e)}")
        return False



def find_leads():
    keywords = [
        "AI tools", "React apps", "Fintech", "eCommerce automation",
        "Voice AI", "Smart Contracts", "Customer Support Automation"
    ]
    topic = random.choice(keywords)
    prompt_seed = str(uuid.uuid4())[:8]  # Random seed to break repetition

    prompt = f"""
    Based on this profile: {profile}

    Find 5 different companies or individuals from tech startups or digital agencies
    working in or around the topic of: "{topic}". These companies should have a
    business need that relates to our profile.

    For each lead, return:
      - Name
      - Role
      - Company
      - Industry
      - Email
      - Why they may need our services

    Only return a JSON list of dictionaries — no markdown, no explanation.
    Only include business emails (no Gmail/Yahoo/Hotmail).
    Ensure diversity and avoid repetition across leads.
    Request ID: {prompt_seed}
    """

    max_retries = 5
    delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            response = Runner.run_sync(agent, input=prompt, run_config=config)
            output = response.final_output.strip()
            if not output:
                raise ValueError("Model returned an empty response.")
            leads = extract_json_from_output(output)
            if not isinstance(leads, list):
                raise ValueError("Expected a JSON list, got something else.")
            return leads
        except Exception as e:
            if "503" in str(e) or "overloaded" in str(e).lower():
                print(f"Model overloaded. Retrying in {delay} seconds (Attempt {attempt}/{max_retries})...")
                time.sleep(delay)
                delay *= 2
                continue
            print(f"Error finding leads: {str(e)}")
            raise e
    raise RuntimeError("Model is still overloaded.")

def generate_email(lead):
    tone = "technical" if lead["Role"].lower() in ["cto", "lead engineer"] else "visionary"
    email_prompt = f"""
Write a cold outreach email to {lead['Role']} {lead['Name']} at {lead['Company']}.

Use this info:
- My Name: Habib ullah
- My Title: Chief Executive Officer
- My Company: XapRise Solutions
- My Contact: 03343295024

Context:
- Mention how we can help with: "{lead['Why they may need our services']}"

Objective:
- Get their attention with a **psychological hook** like:
  "You're losing clients because..." or 
  "We noticed your site is..." or
  "You may be missing out on leads due to..."

Subject:
- Write Email Subject Like This
"YOUR BUSINESS WILL LOSE CLIENTS IF.."

Requirements:
- Tone: {tone}
- Length: 50–80 words max
- No subject line inside the body
- 2–3 short paragraphs
- No self-praise like “we are best”, focus on **their problem**.
- No generic claims — use one **concrete, observed insight**
- Make it feel like we researched their business
- Do NOT list all our services
- Mention **only 1 specific thing** we could help with
- End with a soft, casual CTA like “Worth a quick chat?” or “Open to a quick call?”

Output Format:
Plain text only — use `\n` for line breaks. No HTML, no markdown.
Only output the **email body**, not the subject line.
"""


    response = Runner.run_sync(agent, input=email_prompt, run_config=config)
    return response.final_output.strip()

def send_email(to_email, subject, content, lead):
    url = "https://api.brevo.com/v3/smtp/email"
    escaped_content = html.escape(content)
    paragraphs = escaped_content.split('\n\n')
    html_paragraphs = ''.join('<p>' + para.replace('\n', '<br>') + '</p>' for para in paragraphs)

    headers = {
        "accept": "application/json",
        "api-key": brevo_api_key,
        "content-type": "application/json"
    }

    payload = {
        "sender": {"name": "XapRise Solutions", "email": "codecraftersweb3@gmail.com"},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html_paragraphs
    }

    response = requests.post(url, json=payload, headers=headers)
    print(f"Sent email to {to_email}: {response.status_code}")
    log_email(lead)
    print(f"Logged: {lead['Email']} - {response.status_code}")
    return response.status_code, response.text



def log_email(lead):
    data = {
        "name": lead.get("Name"),
        "company": lead.get("Company"),
        "email": lead.get("Email"),
        "timestamp": datetime.now().isoformat()
    }
    supabase.table("cold_mailing_agent").insert(data).execute()


def show_logs():
    try:
        response = supabase.table("cold_mailing_agent").select("*").execute()
        data = response.data
        if not data:
            print("No outreach logs available.")
        else:
            logs_df = pd.DataFrame(data)
            print(logs_df.to_string(index=False))
    except Exception as e:
        print(f"Error loading logs: {str(e)}")

def run_outreach():
    try:
        leads = find_leads()
        for lead in leads:
            if is_duplicate(lead["Email"]):
                print(f"Skipping duplicate email to: {lead['Email']}")
                continue

            email_text = generate_email(lead)

            status, response = send_email(
                lead['Email'],
                f"Helping {lead['Company']}",
                email_text,
                lead
            )
            if status == 201:
                print(f"Email sent successfully to {lead['Email']}!")
            else:
                print(f"Failed to send email to {lead['Email']}: {response}")
    except Exception as e:
        print(f"Outreach failed: {str(e)}")


schedule.every(72).minutes.do(run_outreach)

while True:
    schedule.run_pending()
    time.sleep(1)

