from dotenv import load_dotenv
import os
import openai
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase

load_dotenv()

openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(
    model_name="mistralai/mistral-7b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)

df = pd.read_csv("amazon-products.csv")

df["desc_len"] = df["description"].astype(str).apply(len)
longest_desc = df.loc[df["desc_len"].idxmax()]
print("Product with Longest Description:\n")
print(longest_desc[["title", "description"]])

engine = create_engine("sqlite:///products.db")
df.to_sql("products", engine, if_exists="replace", index=False)
db = SQLDatabase(engine)

sql_agent = SQLDatabaseChain.from_llm(llm, db=db, verbose=True)

response1 = sql_agent.invoke("List all categories available in the dataset.")
print("\nAll Categories:\n", response1)

response2 = sql_agent.invoke("Which category has the most products?")
print("\nCategory with Most Products:\n", response2)

