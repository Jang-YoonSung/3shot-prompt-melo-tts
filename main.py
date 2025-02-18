from fastapi import FastAPI
from router import db

app = FastAPI()

app.include_router(db.router)