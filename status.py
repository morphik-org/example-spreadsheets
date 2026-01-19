from morphik import Morphik
import os
from dotenv import load_dotenv

load_dotenv()

morphik = Morphik(uri=os.getenv("MORPHIK_URI"))

result = morphik.list_documents().documents

for doc in result:
    print(f"Document {doc.external_id} has status {doc.status['status']}")