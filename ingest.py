from morphik import Morphik
import os
from dotenv import load_dotenv

load_dotenv()

morphik = Morphik(uri=os.getenv("MORPHIK_URI"))

