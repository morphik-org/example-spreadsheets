import os

from dotenv import load_dotenv
from morphik import Morphik
from openai import OpenAI

load_dotenv()

morphik = Morphik(uri=os.getenv("MORPHIK_URI"))
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


