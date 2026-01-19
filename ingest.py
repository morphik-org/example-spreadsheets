from morphik import Morphik
import os
from dotenv import load_dotenv

load_dotenv()

morphik = Morphik(uri=os.getenv("MORPHIK_URI"))

result = morphik.ingest_directory(directory="files")

for doc in result:
    print(f"Ingested document {doc.filename} with id {doc.external_id}. Status: {doc.status['status']}")