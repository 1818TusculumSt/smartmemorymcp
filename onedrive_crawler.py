import os
import msal
import requests
import io
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID")
TENANT_ID = os.getenv("MICROSOFT_TENANT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["Files.Read.All", "User.Read"]

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(
    name=os.getenv("PINECONE_INDEX_NAME"),
    host=os.getenv("PINECONE_HOST")
)

# Initialize local embedding model
print("🧠 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded\n")

def get_access_token():
    """Authenticate using device code flow"""
    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)
    
    flow = app.initiate_device_flow(scopes=SCOPES)
    
    if "user_code" not in flow:
        raise Exception(f"Failed to create device flow: {flow.get('error_description', 'Unknown error')}")
    
    print(f"\n🔐 Go to: {flow['verification_uri']}")
    print(f"📱 Enter code: {flow['user_code']}\n")
    
    result = app.acquire_token_by_device_flow(flow)
    
    if "access_token" in result:
        return result["access_token"]
    else:
        raise Exception(f"Authentication failed: {result.get('error_description')}")

def extract_text_from_file(token, file_item):
    """Download and extract text from supported file types"""
    headers = {"Authorization": f"Bearer {token}"}
    download_url = file_item.get("@microsoft.graph.downloadUrl")
    
    if not download_url:
        return None
    
    # Download file content
    response = requests.get(download_url)
    if response.status_code != 200:
        return None
    
    file_name = file_item["name"].lower()
    content = response.content
    
    try:
        # PDF extraction
        if file_name.endswith('.pdf'):
            pdf = PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        
        # Word doc extraction
        elif file_name.endswith('.docx'):
            doc = Document(io.BytesIO(content))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        
        # Plain text
        elif file_name.endswith('.txt'):
            return content.decode('utf-8', errors='ignore').strip()
        
        else:
            return None
            
    except Exception as e:
        print(f"❌ Failed to extract {file_name}: {e}")
        return None

def upload_to_pinecone(files_data):
    """Upload extracted files to Pinecone with local embeddings"""
    print(f"\n📤 Uploading {len(files_data)} files to Pinecone...")
    
    vectors = []
    for idx, file_data in enumerate(files_data):
        # Generate embedding locally
        text = file_data["text"][:8000]  # Truncate to reasonable length
        embedding = embedding_model.encode(text).tolist()
        
        vector_id = f"doc_{idx}_{file_data['name']}"
        
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "file_name": file_data["name"],
                "file_path": file_data["path"],
                "size": file_data["size"],
                "modified": file_data["modified"],
                "text_preview": text[:500]  # Store preview for display
            }
        })
        print(f"   🔢 Embedded: {file_data['name']}")
    
    # Upsert to Pinecone
    index.upsert(vectors=vectors, namespace="smartdrive")
    print(f"✅ Uploaded {len(vectors)} documents to Pinecone")

def list_documents_folder(token, max_files=50):
    """List files in Documents folder only"""
    headers = {"Authorization": f"Bearer {token}"}
    base_url = "https://graph.microsoft.com/v1.0/me/drive"
    
    # Get Documents folder
    url = f"{base_url}/root:/Documents"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to access Documents folder: {response.text}")
    
    folder_id = response.json()["id"]
    
    # List files in Documents
    url = f"{base_url}/items/{folder_id}/children"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to list files: {response.text}")
    
    items = response.json().get("value", [])
    
    count = 0
    extracted_files = []
    
    for item in items:
        if count >= max_files:
            print(f"\n⚠️ Stopped at {max_files} files (testing limit)")
            break
            
        if "file" in item:  # Only files, not folders
            file_name = item['name']
            print(f"📄 Processing: {file_name}")
            
            text = extract_text_from_file(token, item)
            
            if text:
                print(f"   ✅ Extracted {len(text)} characters")
                extracted_files.append({
                    "name": file_name,
                    "path": item.get("parentReference", {}).get("path", "") + "/" + file_name,
                    "text": text,
                    "size": item.get("size", 0),
                    "modified": item.get("lastModifiedDateTime", "")
                })
            else:
                print(f"   ⚠️ Skipped (unsupported type or failed)")
            
            count += 1
    
    print(f"\n✅ Extracted text from {len(extracted_files)} files")
    return extracted_files

if __name__ == "__main__":
    token = get_access_token()
    print("✅ Authentication successful!\n")
    print("📂 Scanning Documents folder...\n")
    files = list_documents_folder(token, max_files=50)
    
    if files:
        upload_to_pinecone(files)
