import os
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app import index_helpers
from app.index_helpers import init_index, query_index, get_index, insert_index, create_index

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/create_index/{name}")
def create_new_index(name: str):
    create_index(name)
    response = f"Index `{name}` created"
    return response

@app.get("/init_index/{name}")
def start_index(name: str):
    init_index(name)
    response = f"Index `{name}` is initialized"
    return response

@app.get("/read_index")
def read_index():
    return str(get_index())

@app.get("/list_indices")
def list_indices():
    return index_helpers.list_indices()

@app.delete("/delete_index/{doc_id}")
def delete_index(doc_id: str):
    return index_helpers.delete_index(doc_id)

@app.get("/")
def read_root():
    return "API is up"

@app.get("/query/{chat_id}")
def query(text: str, chat_id: str):
    return {"content": query_index(text, chat_id)}

@app.get("/get_chat/{chat_id}")
def get_chat(chat_id: str):
    return index_helpers.get_chat_history(chat_id)

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    filepath = os.path.join("./app/tmp", os.path.basename(file.filename)) # type: ignore
    await save_file_to_disc(file, filepath)

    if insert_index(filepath) is None:
        return "Failed to insert to index"

    if os.path.exists(filepath):
        os.remove(filepath)

    return {"filename": file.filename}

@app.delete("/delete_all_indices")
def delete_all_indices():
    return index_helpers.delete_all_indices()

async def save_file_to_disc(file: UploadFile, filepath: str):
    with open(filepath, "wb") as f:
        f.write(await file.read())
