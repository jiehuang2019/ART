#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-15T21:09:00.713Z
"""


#@title Email Search Code

import httpx

import os
import random
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from textwrap import dedent
from typing import List, Literal, Optional

from datasets import Dataset, Features, Sequence, Value, load_dataset
from pydantic import BaseModel, Field
from tqdm import tqdm

# Training configuration
from src.art.utils import iterate_dataset
from src.art.langgraph import wrap_rollout
import os, glob, shutil, time

import art
from src.art.local import LocalBackend
from src.art.rewards import ruler_score_group
from src.art.skypilot import SkyPilotBackend

import uuid

import weave
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from litellm import acompletion
from tenacity import retry, stop_after_attempt
from src.art.langgraph import init_chat_model

from dotenv import load_dotenv
import asyncio

# Execute "source .venv/bin/activate" before running the python file

random.seed(42)

# Global variables
model = None
backend = None
# Database configuration
DB_PATH = "./enron_emails.db"
EMAIL_DATASET_REPO_ID = "corbt/enron-emails"
SCENARIO_DATASET_REPO_ID = "corbt/enron_emails_sample_questions"

# Global database connection
db_conn = None




# Email and Scenario data models
class Email(BaseModel):
    message_id: str
    date: str  # ISO 8601 string 'YYYY-MM-DD HH:MM:SS'
    subject: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: List[str] = []  # Populated from recipients table
    cc_addresses: List[str] = []  # Populated from recipients table
    bcc_addresses: List[str] = []  # Populated from recipients table
    body: Optional[str] = None
    file_name: Optional[str] = None


class Scenario(BaseModel):
    id: int
    question: str
    answer: str
    message_ids: List[str]  # message_ids (strings) of referenced emails
    how_realistic: float
    inbox_address: str
    query_date: str
    split: Literal["train", "test"]


@dataclass
class SearchResult:
    message_id: str
    snippet: str


class FinalAnswer(BaseModel):
    answer: str
    source_ids: list[str]




def create_email_database():
    """Create the email database from Hugging Face dataset"""
    print("Creating email database from Hugging Face dataset...")
    print(
        "This will download and process the full Enron email dataset - this may take several minutes..."
    )

    # Database schema
    SQL_CREATE_TABLES = """
    DROP TABLE IF EXISTS recipients;
    DROP TABLE IF EXISTS emails_fts;
    DROP TABLE IF EXISTS emails;

    CREATE TABLE emails (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id TEXT UNIQUE,
        subject TEXT,
        from_address TEXT,
        date TEXT,
        body TEXT,
        file_name TEXT
    );

    CREATE TABLE recipients (
        email_id TEXT,
        recipient_address TEXT,
        recipient_type TEXT
    );
    """

    SQL_CREATE_INDEXES_TRIGGERS = """
    CREATE INDEX idx_emails_from ON emails(from_address);
    CREATE INDEX idx_emails_date ON emails(date);
    CREATE INDEX idx_emails_message_id ON emails(message_id);
    CREATE INDEX idx_recipients_address ON recipients(recipient_address);
    CREATE INDEX idx_recipients_type ON recipients(recipient_type);
    CREATE INDEX idx_recipients_email_id ON recipients(email_id);
    CREATE INDEX idx_recipients_address_email ON recipients(recipient_address, email_id);

    CREATE VIRTUAL TABLE emails_fts USING fts5(
        subject,
        body,
        content='emails',
        content_rowid='id'
    );

    CREATE TRIGGER emails_ai AFTER INSERT ON emails BEGIN
        INSERT INTO emails_fts (rowid, subject, body)
        VALUES (new.id, new.subject, new.body);
    END;

    CREATE TRIGGER emails_ad AFTER DELETE ON emails BEGIN
        DELETE FROM emails_fts WHERE rowid=old.id;
    END;

    CREATE TRIGGER emails_au AFTER UPDATE ON emails BEGIN
        UPDATE emails_fts SET subject=new.subject, body=new.body WHERE rowid=old.id;
    END;
    """

    # Create database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_TABLES)
    conn.commit()

    # Load dataset
    print("Loading full email dataset...")
    expected_features = Features(
        {
            "message_id": Value("string"),
            "subject": Value("string"),
            "from": Value("string"),
            "to": Sequence(Value("string")),
            "cc": Sequence(Value("string")),
            "bcc": Sequence(Value("string")),
            "date": Value("timestamp[us]"),
            "body": Value("string"),
            "file_name": Value("string"),
        }
    )
    # Reading the training dataset
    dataset = load_dataset(
        EMAIL_DATASET_REPO_ID, features=expected_features, split="train"
    )
    print(f"Dataset contains {len(dataset)} total emails")

    # Populate database with ALL emails (not limited to 1000)
    print("Populating database with all emails...")
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("BEGIN TRANSACTION;")

    record_count = 0
    skipped_count = 0
    duplicate_count = 0
    processed_emails = set()  # Track (subject, body, from) tuples for deduplication

    #tqdm: progress bar wrapper
    for email_data in tqdm(dataset, desc="Inserting emails"):
        message_id = email_data["message_id"]
        subject = email_data["subject"]
        from_address = email_data["from"]
        date_obj: datetime = email_data["date"]
        body = email_data["body"]
        file_name = email_data["file_name"]
        to_list = [str(addr) for addr in email_data["to"] if addr]
        cc_list = [str(addr) for addr in email_data["cc"] if addr]
        bcc_list = [str(addr) for addr in email_data["bcc"] if addr]

        # Apply the same filters as the original project
        total_recipients = len(to_list) + len(cc_list) + len(bcc_list)

        # Filter out very long emails and those with too many recipients
        if len(body) > 5000:
            skipped_count += 1
            continue

        if total_recipients > 30:
            skipped_count += 1
            continue

        # Deduplication check (same as original project)
        email_key = (subject, body, from_address)
        if email_key in processed_emails:
            duplicate_count += 1
            continue
        else:
            processed_emails.add(email_key)

        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
            INSERT INTO emails (message_id, subject, from_address, date, body, file_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (message_id, subject, from_address, date_str, body, file_name),
        )

        # Insert recipients
        recipient_data = []
        for addr in to_list:
            recipient_data.append((message_id, addr, "to"))
        for addr in cc_list:
            recipient_data.append((message_id, addr, "cc"))
        for addr in bcc_list:
            recipient_data.append((message_id, addr, "bcc"))

        if recipient_data:
            cursor.executemany(
                """
                INSERT INTO recipients (email_id, recipient_address, recipient_type)
                VALUES (?, ?, ?)
            """,
                recipient_data,
            )

        record_count += 1

    conn.commit()

    # Create indexes and triggers
    print("Creating indexes and FTS...")
    cursor.executescript(SQL_CREATE_INDEXES_TRIGGERS)
    cursor.execute('INSERT INTO emails_fts(emails_fts) VALUES("rebuild")')
    conn.commit()

    print(f"Successfully created database with {record_count} emails.")
    print(f"Skipped {skipped_count} emails due to length/recipient limits.")
    print(f"Skipped {duplicate_count} duplicate emails.")
    return conn


def get_db_connection():
    """Get database connection"""
    if os.path.exists(DB_PATH):
        print(f"Loading existing database from {DB_PATH}")
        db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    else:
        db_conn = create_email_database()
    return db_conn


def search_emails(
    inbox: str,
    keywords: List[str],
    from_addr: Optional[str] = None,
    to_addr: Optional[str] = None,
    sent_after: Optional[str] = None,
    sent_before: Optional[str] = None,
    max_results: int = 10,
) -> List[SearchResult]:
    """Search the email database based on keywords and filters"""
    conn = get_db_connection()
    cursor = conn.cursor()

    where_clauses: List[str] = []
    params: List[str | int] = []

    if not keywords:
        raise ValueError("No keywords provided for search.")

    if max_results > 10:
        raise ValueError("max_results must be less than or equal to 10.")

    # FTS5 default is AND, so just join keywords. Escape quotes for safety.
    fts_query = " ".join(f""" "{k.replace('"', '""')}" """ for k in keywords)
    where_clauses.append("fts.emails_fts MATCH ?")
    params.append(fts_query)

    # Inbox filter
    where_clauses.append("""
        (e.from_address = ? OR EXISTS (
            SELECT 1 FROM recipients r_inbox
            WHERE r_inbox.recipient_address = ? AND r_inbox.email_id = e.message_id
        ))
    """)
    params.extend([inbox, inbox])

    if from_addr:
        where_clauses.append("e.from_address = ?")
        params.append(from_addr)

    if to_addr:
        where_clauses.append("""
            EXISTS (
                SELECT 1 FROM recipients r_to
                WHERE r_to.recipient_address = ? AND r_to.email_id = e.message_id
            )
        """)
        params.append(to_addr)

    if sent_after:
        where_clauses.append("e.date >= ?")
        params.append(f"{sent_after} 00:00:00")

    if sent_before:
        where_clauses.append("e.date < ?")
        params.append(f"{sent_before} 00:00:00")

    sql = f"""
        SELECT
            e.message_id,
            snippet(emails_fts, -1, '<b>', '</b>', ' ... ', 15) as snippet
        FROM
            emails e JOIN emails_fts fts ON e.id = fts.rowid
        WHERE
            {" AND ".join(where_clauses)}
        ORDER BY
            e.date DESC
        LIMIT ?;
    """
    params.append(max_results)

    cursor.execute(sql, params)
    results = cursor.fetchall()

    return [SearchResult(message_id=row[0], snippet=row[1]) for row in results]


def read_email(message_id: str) -> Optional[Email]:
    """Retrieve a single email by its message_id"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get email details
    cursor.execute(
        "SELECT message_id, date, subject, from_address, body, file_name FROM emails WHERE message_id = ?",
        (message_id,),
    )
    email_row = cursor.fetchone()

    if not email_row:
        return None

    msg_id, date, subject, from_addr, body, file_name = email_row

    # Get recipients
    cursor.execute(
        "SELECT recipient_address, recipient_type FROM recipients WHERE email_id = ?",
        (message_id,),
    )
    recipient_rows = cursor.fetchall()

    to_addresses = []
    cc_addresses = []
    bcc_addresses = []

    for addr, type_val in recipient_rows:
        if type_val.lower() == "to":
            to_addresses.append(addr)
        elif type_val.lower() == "cc":
            cc_addresses.append(addr)
        elif type_val.lower() == "bcc":
            bcc_addresses.append(addr)

    return Email(
        message_id=msg_id,
        date=date,
        subject=subject,
        from_address=from_addr,
        to_addresses=to_addresses,
        cc_addresses=cc_addresses,
        bcc_addresses=bcc_addresses,
        body=body,
        file_name=file_name,
    )


def load_training_scenarios(
    split: Literal["train", "test"] = "train",
    limit: Optional[int] = None,
    max_messages: Optional[int] = 1,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[Scenario]:
    """Load training scenarios from Hugging Face dataset"""
    print(f"Loading {split} scenarios from Hugging Face...")
    dataset: Dataset = load_dataset(SCENARIO_DATASET_REPO_ID, split=split)

    if max_messages is not None:
        dataset = dataset.filter(lambda x: len(x["message_ids"]) <= max_messages)

    if shuffle or (seed is not None):
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        else:
            dataset = dataset.shuffle()

    # Convert each row to a Scenario object
    scenarios = [Scenario(**row, split=split) for row in dataset]

    if max_messages is not None:
        scenarios = [s for s in scenarios if len(s.message_ids) <= max_messages]

    if shuffle:
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(scenarios)
        else:
            random.shuffle(scenarios)
    # Why do both max_messages and limit exist?
    if limit is not None:
        scenarios = scenarios[:limit]

    print(f"Loaded {len(scenarios)} scenarios.")
    return scenarios



# To train this email search agent using LangGraph, click **Runtime** > **Run all**. Make sure you've enabled a free Tesla T4 GPU!
# 
# <div class="align-center">
# <a href="https://github.com/openpipe/art"><img src="https://github.com/openpipe/art/raw/main/assets/ART_pill.png" height="50"></a>
# <a href="https://discord.gg/zbBHRUpwf4"><img src="https://github.com/openpipe/art/raw/main/assets/Discord.png" height="50"></a>
# <a href="https://art.openpipe.ai"><img src="https://github.com/openpipe/art/raw/main/assets/Documentation_pill.png" height="50"></a>
# 
# Questions? Join the Discord and ask away! For feature requests or to leave a star, visit our [Github](https://github.com/openpipe/art).
# 
# </div>
# 
# <a href="https://art.openpipe.ai/"><img src="https://github.com/openpipe/art/raw/main/assets/Header_separator.png" height="5"></a>
# 
# **Email Search Agent with LangGraph**
# 
# In this notebook, you will be using [ART](https://github.com/openpipe/art) together with [LangGraph](https://langchain-ai.github.io/langgraph/) to train your own ART•E agent from scratch! This implementation demonstrates how to integrate LangGraph's agent framework with ART's training capabilities.
# 
# Beginning with a Qwen 2.5 7B base model, you will train it to search through emails and answer questions about them using LangGraph's ReAct agent pattern. You will construct an [agentic environment](#Environment), define a [rollout](#Rollout) using LangGraph, and run a [training loop](#Loop). You will also learn how to use [RULER](#ruler) to judge the quality of the agent's answers.
# 
# **RULER**
# 
# RULER is a robust technique for evaluating the quality of an agent's answers and training the agent to produce more of its best completions. To learn more about RULER, see the [RULER documentation](https://art.openpipe.ai/fundamentals/ruler).
# 
# Now let's get started!


#@title 💿 Installation

# Portions adapted from Unsloth Notebooks (https://github.com/unslothai/notebooks)
# Copyright (c) Unsloth contributors.
# License: GNU LGPL v3.0.
# Modifications by OpenPipe:
# - switched to uv
# - changed vllm/triton pinning logic
# - added litellm/protobuf pins
# See /licenses/LGPL-3.0.txt and /licenses/GPL-3.0.txt for full text.

# =============================================================================
# import os
# 
# !uv pip install openpipe-art[backend,langgraph]==0.4.11 langchain-core langgraph langchain_openai tenacity datasets  --prerelease allow --no-cache-dir
# if "COLAB_" not in "".join(os.environ.keys()):
#     !uv pip install openpipe-art[backend,langgraph]==0.4.11 langchain-core langgraph langchain_openai tenacity datasets  --prerelease allow --no-cache-dir
# else:
#     try:
#         import numpy
# 
#         get_numpy = f"numpy=={numpy.__version__}"
#     except:
#         get_numpy = "numpy"
#     try:
#         import subprocess
# 
#         is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
#     except:
#         is_t4 = False
#     get_vllm, get_triton = (
#         ("vllm==0.9.2", "triton==3.2.0") if is_t4 else ("vllm", "triton")
#     )
#     !uv pip install --upgrade \
#         openpipe-art[backend,langgraph]==0.4.11 langchain-core langgraph langchain_openai tenacity datasets protobuf==5.29.5 {get_vllm} {get_numpy} wandb==0.16.6 --prerelease allow --no-cache-dir
#     # !uv pip install --upgrade \
#     #     openpipe-art[backend,langgraph]==0.4.11 langchain-core langgraph langchain_openai tenacity datasets protobuf==5.29.5 {get_vllm} {get_numpy} --prerelease allow --no-cache-dir
#     !uv pip install -qqq {get_triton}
# =============================================================================

# <a name="Environment-Variables"></a>
# 
# ### Environment Variables
# 
# **OpenAI (used for RULER judge model)**
# 
# Our RULER reward function queries third-party models to judge the quality of the agent's performance. Any model supported by LiteLLM works. For this example we'll use OpenAI's o4-mini model, so we'll need to set the `OPENAI_API_KEY` environment variable.
# 
# **Weights & Biases (optional)**
# 
# Later on in the notebook, we'll be creating a model that can automatically logs metrics to Weights & Biases and chat completions to Weave. In order to do so, you'll need to provide your Weights & Biases API key as an environment variable.


load_dotenv()

# Required

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is required for RULER functionality when using openai/o4-mini."
    )

# Optional
# os.environ["WANDB_API_KEY"] = "YOUR_API_KEY"

if not os.environ.get("WANDB_API_KEY"):
    print("WANDB_API_KEY is not set. We'll skip logging metrics to Weights & Biases.")

# <a name="Environment"></a>
# 
# ### Email Search Environment
# 
# ART allows your agent to learn by interacting with its environment. In this example, we'll create an environment where the agent can search through emails and answer questions about them using LangGraph's tools integration.
# 
# The agent will have access to three tools:
# 
# 1. `search_inbox` - Search for emails by keywords
# 2. `read_email` - Read a specific email by message ID
# 3. `return_final_answer` - Return the final answer with source email IDs


# ### Creating a Model
# 
# Now that we've defined the rules of our environment, we can create a model that will learn to search emails effectively. We'll use a Qwen 2.5 7B model for this example.


#drive.mount('/content/gdrive')  # Mount Google Drive

# Destination path in your Google Drive
#SAVE_DIR = "/content/gdrive/MyDrive/MeekAI/SAVE/"

# Ensure the destination directory exists
#os.makedirs(os.path.dirname(SAVE_DIR), exist_ok=True)

async def initialize_model(which_gpu = "RTX 5090"):
    model_name=  "email-agent-langgraph-001"
    project_name = "email-search-agent-langgraph"
    # Declare the model
    global model
    model = art.TrainableModel(
        name = model_name,
        project = project_name,
        base_model="Qwen/Qwen2.5-7B-Instruct",
        #base_model = os.path.join(SAVE_DIR, ".art", project_name, "models", model_name, "checkpoints", "0003")
    )
    
    # To run on a T4, we need to override some config defaults.
    # =============================================================================
    # model._internal_config = art.dev.InternalModelConfig(
    #     init_args=art.dev.InitArgs(
    #         max_seq_length=8192,
    #     ),
    #     engine_args=art.dev.EngineArgs(
    #         enforce_eager=True,
    #         gpu_memory_utilization=0.8,
    #     ),
    # )
    # =============================================================================
    
    # Initialize the server
# =============================================================================
#     backend = LocalBackend(
#         # Normally we don't want to run the server in-process, but for the output
#         # to show up properly on Google Colab we'll enable this.
#         in_process=True,
#         path="./.art",
#         #path=os.path.join(SAVE_DIR, "./.art"),
#     )
# =============================================================================
    project_root = os.getcwd()
    global backend
    backend = await SkyPilotBackend.initialize_cluster(
        # name of the cluster in SkyPilot's registry
        cluster_name="meekAI_cluster",
        # version of openpipe-art that should be installed on the remote cluster
        # default to version installed on the client
        art_version="0.5.2",
        #art_version=project_root,
        # path to environment variables (e.g. WANDB_API_KEY) to set on the remote cluster
        env_path=".env",
        # the GPU the cluster is equipped with
        gpu=which_gpu
        # alternatively, more complicated requirements can be specified in
        # the `resources` argument
    )
    
    # Register the model with the local Backend (sets up logging, inference, and training)
    await model.register(backend)
    print("Model created and registered with the backend")
# <a name="Rollout"></a>
# 
# ### Defining a Rollout with LangGraph
# 
# A rollout is a single episode of an agent performing its task. In this example, we'll use LangGraph's ReAct agent to handle the rollout. The rollout function presents the agent with an email search scenario, and the LangGraph agent uses the available tools to search for emails and answer the question.
# 
# When the agent provides a final answer, the `correct` metric is calculated based on whether the answer is correct.



if os.getenv("WANDB_API_KEY", ""):
    weave.init(model.project, settings={"print_call_link": False})

MAX_TURNS = 20

class CorrectnessJudgeResponse(BaseModel):
    reasoning: str = Field(description="Explanation of the reasoning process.")
    accept: bool = Field(description="Whether the AI answer should be accepted.")


@retry(stop=stop_after_attempt(3))
async def judge_correctness(
    scenario: Scenario, answer: str
) -> CorrectnessJudgeResponse:
    system_prompt = dedent(
        """
        You are given a question, the reference answer (labelled **Reference answer**), and an answer generated by an AI assistant (labelled **AI answer**).

        Your task is to decide whether the AI answer is correct and should be accepted. You should accept the answer if it contains the relevant information from the reference answer. You should not accept the answer if it is missing information relevant to the question, or if it contradicts the reference answer.
        """
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Question: {scenario.question}\n"
                f"Reference answer: {scenario.answer}\n"
                f"AI answer: {answer}"
            ),
        },
    ]

    response = await acompletion(
        model="openai/gpt-4.1",
        messages=messages,
        response_format=CorrectnessJudgeResponse,
    )

    first_choice = response.choices[0]
    raw_content = first_choice.message.content or "{}"

    try:
        return CorrectnessJudgeResponse.model_validate_json(raw_content)
    except Exception as e:
        return CorrectnessJudgeResponse(
            reasoning=f"Parse error: {e}\nRaw: {raw_content}", accept=False
        )


class ProjectTrajectory(art.Trajectory):
    final_answer: FinalAnswer | None = None


class EmailScenario(BaseModel):
    step: int
    scenario: Scenario


@weave.op
async def rollout(model: art.Model, email_scenario: EmailScenario) -> ProjectTrajectory:
    scenario = email_scenario.scenario

    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "scenario_id": scenario.id,
            "step": email_scenario.step,
        },
    )

    system_prompt = dedent(
        f"""
        You are an email search agent. You are given a user query and a list of tools you can use to search the user's email. Use the tools to search the user's emails and find the answer to the user's query. You may take up to {MAX_TURNS} turns to find the answer, so if your first search doesn't find the answer, you can try with different keywords.

        User's email address is {scenario.inbox_address}
        Today's date is {scenario.query_date}

        When you have found the answer, use the return_final_answer_tool to provide your final answer along with the source message IDs.
        """
    )

    # Store final answer in trajectory
    final_answer = None

    # Define tools inside the rollout function to access local variables
    @tool
    def search_inbox_tool(keywords: list[str]) -> list[dict]:
        """Search the inbox for emails matching the given keywords and return
        a list of dictionaries so the LLM can easily consume them."""
        results = search_emails(
            inbox=scenario.inbox_address,
            keywords=keywords,
            sent_before=scenario.query_date,
        )
        return [asdict(result) for result in results]

    @tool
    def read_email_tool(message_id: str) -> dict | None:
        """Read a specific email by message ID."""
        email = read_email(message_id)
        if email:
            return email.model_dump() # Converting to a dict
        return None

    @tool
    def return_final_answer_tool(answer: str, reference_message_ids: list[str]) -> dict:
        """Return the final answer and the message IDs of the emails that were used to generate the answer."""
        nonlocal final_answer
        final_answer = FinalAnswer(answer=answer, source_ids=reference_message_ids)
        return final_answer.model_dump()

    # Create LangGraph tools
    tools = [search_inbox_tool, read_email_tool, return_final_answer_tool]
    # Temperature controls the randomness or creativity generated by LLMs during inference. 1.0 is medium, a balanced output that is both coherent and has some variability
    chat_model = init_chat_model(model.name, temperature=1.0)
    # Create the LangGraph ReAct agent
    react_agent = create_react_agent(chat_model, tools)

    try:
        # Run the agent
        config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": MAX_TURNS,
        }

        await react_agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=scenario.question),
                ]
            },
            config=config,
        )

        # Check if we got a final answer
        if final_answer:
            traj.final_answer = final_answer
            # Score the trajectory
            correctness_judge_response = await judge_correctness(
                scenario, traj.final_answer.answer
            )
            # Would this be strictly 0 and 1?
            traj.metrics["correct"] = float(correctness_judge_response.accept)

    except Exception as e:
        print(f"Error running LangGraph agent: {e}")
        # Add error information to trajectory
        traj.messages_and_choices.append(
            {"role": "assistant", "content": f"Error: {str(e)}"}
        )

    return traj


print("LangGraph rollout function defined!")

# <a name="ruler"></a>
# 
# ### How RULER works
# 
# **RULER** leverages two key insights:
# 
# 1. Relative scoring is easier than absolute scoring: It's easier for an LLM to rank several solutions relative to each other than to score them in isolation
# 2. GRPO only needs relative scores: Since GRPO normalizes scores within each group, only the relative rankings matter, not absolute values
# 
# The process:
# 
# 1. Generate N trajectories for a given scenario
# 2. Pass all N trajectories to **RULER**
# 3. **RULER** deduplicates common prefixes (e.g., identical system messages)
# 4. An LLM judge scores each trajectory from 0 to 1 based on goal achievement
# 5. These scores are used directly as rewards in GRPO training
# 
# To learn more about **RULER**, check out the [RULER docs](https://art.openpipe.ai/fundamentals/ruler).


# =============================================================================
# #@title Sample RULER evaluation
# 
# import art
# from art.rewards import ruler_score_group
# 
# # Test RULER with a simple example
# base_messages = [
#     {"role": "system", "content": "You count numbers using numeric symbols."},
#     {"role": "user", "content": "Count to 10."},
# ]
# 
# good_trajectory = art.Trajectory(
#     messages_and_choices=[
#         *base_messages,
#         {"role": "assistant", "content": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"},
#     ],
#     reward=0,
# )
# 
# mediocre_trajectory = art.Trajectory(
#     messages_and_choices=[
#         *base_messages,
#         {
#             "role": "assistant",
#             "content": "one, two, three, four, five, six, seven, eight, nine, ten",
#         },
#     ],
#     reward=0,
# )
# 
# bad_trajectory = art.Trajectory(
#     messages_and_choices=[
#         *base_messages,
#         {"role": "assistant", "content": "a, b, c, d, e, f, g, h, i, j"},
#     ],
#     reward=0,
# )
# 
# sample_group = art.TrajectoryGroup(
#     trajectories=[
#         good_trajectory,
#         mediocre_trajectory,
#         bad_trajectory,
#     ]
# )
# 
# judged_group = await ruler_score_group(sample_group, "openai/o4-mini", debug=True)
# assert judged_group is not None
# 
# # Display rankings
# sorted_trajectories = sorted(
#     judged_group.trajectories, key=lambda t: t.reward, reverse=True
# )
# for rank, traj in enumerate(sorted_trajectories, 1):
#     messages = traj.messages()
#     print(f"\nRank {rank}: Score {traj.reward:.3f}")
#     print(f"  Response: {messages[-1]['content'][:50]}...")
# 
# 
# 
# #@title My Test RULER evaluation 1: Historical Fact
# 
# # Test RULER with a simple example
# base_messages = [
#     {"role": "system", "content": "You tell historical facts."},
#     {"role": "user", "content": "Who is the first president of the United States?"},
# ]
# 
# good_trajectory = art.Trajectory(
#     messages_and_choices=[
#         *base_messages,
#         # Score = 1.0 for correctness
#         {"role": "assistant", "content": "George Washington"},
#     ],
#     reward=0,
# )
# 
# mediocre_trajectory = art.Trajectory(
#     messages_and_choices=[
#         *base_messages,
#         {
#             "role": "assistant",
#             # Score = 0.5 for having the correct last name but incomplete
#             "content": "Washington",
#         },
#     ],
#     reward=0,
# )
# 
# bad_trajectory = art.Trajectory(
#     messages_and_choices=[
#         *base_messages,
#         # Score = 0 for being completely incorrect
#         {"role": "assistant", "content": "Abraham Lincoln"},
#     ],
#     reward=0,
# )
# 
# sample_group = art.TrajectoryGroup(
#     trajectories=[
#         good_trajectory,
#         mediocre_trajectory,
#         bad_trajectory,
#     ]
# )
# 
# judged_group = await ruler_score_group(sample_group, "openai/o4-mini", debug=True)
# assert judged_group is not None
# 
# # Display rankings
# sorted_trajectories = sorted(
#     judged_group.trajectories, key=lambda t: t.reward, reverse=True
# )
# for rank, traj in enumerate(sorted_trajectories, 1):
#     messages = traj.messages()
#     print(f"\nRank {rank}: Score {traj.reward:.3f}")
#     print(f"  Response: {messages[-1]['content'][:50]}...")
# 
# 
# 
# #@title My Test RULER evaluation 2: Historical Facts, not so clear cut #1
# 
# # Test RULER with a simple example
# base_messages = [
#     {"role": "system", "content": "You tell historical facts."},
#     {"role": "user", "content": "What caused the Civil War in USA?"},
# ]
# 
# good_trajectory = art.Trajectory(
#     messages_and_choices=[
#         *base_messages,
#         # Score = 0.6 for pointing out the primary reason but lacks context
#         {"role": "assistant", "content": "Slavery"},
#     ],
#     reward=0,
# )
# 
# mediocre_trajectory = art.Trajectory(
#     messages_and_choices=[
#         *base_messages,
#         {
#             "role": "assistant",
#             # Score = 0.5 for mentioning part of the cause but not including slavery as the primary reason
#             "content": "Economic Differences",
#         },
#     ],
#     reward=0,
# )
# 
# bad_trajectory = art.Trajectory(
#     messages_and_choices=[
#         *base_messages,
#         # Score = 0.1 for over-simplifying the causes as conflicts between political parties
#         {"role": "assistant", "content": "Democrats vs Republicans"},
#     ],
#     reward=0,
# )
# 
# sample_group = art.TrajectoryGroup(
#     trajectories=[
#         good_trajectory,
#         mediocre_trajectory,
#         bad_trajectory,
#     ]
# )
# 
# judged_group = await ruler_score_group(sample_group, "openai/o4-mini", debug=True)
# assert judged_group is not None
# 
# # Display rankings
# sorted_trajectories = sorted(
#     judged_group.trajectories, key=lambda t: t.reward, reverse=True
# )
# for rank, traj in enumerate(sorted_trajectories, 1):
#     messages = traj.messages()
#     print(f"\nRank {rank}: Score {traj.reward:.3f}")
#     print(f"  Response: {messages[-1]['content'][:50]}...")
# 
# # 
# 
# 
# #@title My Test RULER evaluation 3: Historical Facts, not so clear cut #2
# 
# # Test RULER with a simple example
# base_messages = [
#     {"role": "system", "content": "You tell historical facts."},
#     {"role": "user", "content": "What caused the Civil War in USA?"},
# ]
# 
# good_trajectory = art.Trajectory(
#     messages_and_choices=[
#         *base_messages,
#         # Score = 0.7 for pointing out secondary cause and some context
#         {"role": "assistant", "content": "Economic Differences between the Industrial North and agricultural South"},
#     ],
#     reward=0,
# )
# 
# mediocre_trajectory = art.Trajectory(
#     messages_and_choices=[
#         *base_messages,
#         {
#             "role": "assistant",
#             # Score = 0.9 for pointing out the primary cause and both economic and moral context
#             "content": "Economic and moral conflict from slavery",
#         },
#     ],
#     reward=0,
# )
# 
# bad_trajectory = art.Trajectory(
#     messages_and_choices=[
#         *base_messages,
#         # Score = 0.3 for pointing out a minor cause, but failed to mention slavery
#         {"role": "assistant", "content": "Federal vs. state rights"},
#     ],
#     reward=0,
# )
# 
# sample_group = art.TrajectoryGroup(
#     trajectories=[
#         good_trajectory,
#         mediocre_trajectory,
#         bad_trajectory,
#     ]
# )
# 
# judged_group = await ruler_score_group(sample_group, "openai/o4-mini", debug=True)
# assert judged_group is not None
# 
# # Display rankings
# sorted_trajectories = sorted(
#     judged_group.trajectories, key=lambda t: t.reward, reverse=True
# )
# for rank, traj in enumerate(sorted_trajectories, 1):
#     messages = traj.messages()
#     print(f"\nRank {rank}: Score {traj.reward:.3f}")
#     print(f"  Response: {messages[-1]['content'][:50]}...")
# =============================================================================

# <a name="Loop"></a>
# 
# ### Training Loop with LangGraph
# 
# The training loop is where the magic happens. For each of the steps defined below, the rollout function will be called multiple times in parallel using LangGraph's ReAct agent. Each scenario will produce a trajectory, which will be used to update the model.
# 
# The `gather` step will wait for all of the trajectories to be generated, then it will use RULER to assign relative scores to each trajectory.
# 
# Our notebook will then delete all but the most recent checkpoint and train the model on the scored trajectories.
from openai.types.chat.chat_completion import Choice as ChatChoice, ChatCompletionMessage

def validate_trajectories(groups):
    for gi, group in enumerate(groups):
        for ti, traj in enumerate(group.trajectories):
            if isinstance(traj, Exception):
                print(f"[TRAJ ERROR] group {gi}, traj {ti} is Exception: {traj}")
                continue

            mac = getattr(traj, "messages_and_choices", None)
            if mac is None:
                print(f"[WARN] group {gi}, traj {ti} has no messages_and_choices attr: {type(traj)}")
                continue

            print(f"[INFO] group {gi}, traj {ti}, {len(mac)} messages_and_choices")
            for mi, msg in enumerate(list(mac)):  # 用 list(mac) 避免迭代时修改列表问题

                # 1) 如果是 OpenAI Choice 对象，转成标准 dict
                if isinstance(msg, ChatChoice):
                    m: ChatCompletionMessage = msg.message
                    new_msg = {
                        "role": m.role,                       # 'assistant'
                        "content": m.content or "",
                    }

                    # 如果有 tool_calls，则一并保留（结构简化一点即可）
                    if m.tool_calls:
                        new_msg["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                            for tc in m.tool_calls
                        ]

                    # ⚠️ 对于 tool_calls 类型的 step，通常不需要参与 GRPO 的 logprobs 计算，
                    #    所以这里我刻意不把 msg.logprobs 塞进去，避免 ART 去优化这一段。
                    #    如果你以后想训练这段，可以再专门设计映射。
                    mac[mi] = new_msg
                    msg = new_msg  # 下面的检查继续用新 dict
                    # 继续往下走，不 raise
                    # continue  # 如果你不想再做其它检查，这里可以直接 continue

                # 2) 非 dict 但也不是 Choice，先打印再抛错（便于发现其它类型）
                if not isinstance(msg, dict):
                    print(f"[BAD MSG] g{gi} t{ti} m{mi}: not dict -> {type(msg)} {msg}")
                    raise RuntimeError("Non-dict message in messages_and_choices")

                # 3) dict 但缺 role -> 这是会在后端炸 KeyError 的情况
                if "role" not in msg:
                    print(f"[BAD MSG] g{gi} t{ti} m{mi}: missing role -> {msg}")
                    raise RuntimeError("Message without role in messages_and_choices")

                # 4) 有 logprobs 但 role 不是 assistant，给个告警
                if "logprobs" in msg and msg.get("role") != "assistant":
                    print(f"[WARN] g{gi} t{ti} m{mi}: logprobs but role={msg.get('role')} -> {msg}")

def validate_trajectories_v3(groups):
    for gi, group in enumerate(groups):
        for ti, traj in enumerate(group.trajectories):
            if isinstance(traj, Exception):
                print(f"[TRAJ ERROR] group {gi}, traj {ti} is Exception: {traj}")
                continue

            mac = getattr(traj, "messages_and_choices", None)
            if mac is None:
                print(f"[WARN] group {gi}, traj {ti} has no messages_and_choices attr: {type(traj)}")
                continue

            print(f"[INFO] group {gi}, traj {ti}, {len(mac)} messages_and_choices")
            for mi, msg in enumerate(mac):
                if not isinstance(msg, dict):
                    print(f"[BAD MSG] g{gi} t{ti} m{mi}: not dict -> {type(msg)} {msg}")
                    raise RuntimeError("Non-dict message in messages_and_choices")

                if "role" not in msg:
                    print(f"[BAD MSG] g{gi} t{ti} m{mi}: missing role -> {msg}")
                    raise RuntimeError("Message without role in messages_and_choices")

                if "logprobs" in msg and msg.get("role") != "assistant":
                    print(f"[WARN] g{gi} t{ti} m{mi}: logprobs but role={msg.get('role')} -> {msg}")

async def train() :

    training_config = {
        "groups_per_step": 2,
        "num_epochs": 1,  # Default is 20
        "rollouts_per_group": 4,
        "learning_rate": 1e-5,  # What is this?
        "max_steps": 2,  # Default is 20
    }
    
    # =============================================================================
    # def newest_adapter_dir(search_roots=("/content", "/root", "/home")):
    #     candidates = []
    #     for root in search_roots:
    #         # look for adapter_config.json to identify LoRA adapter dirs
    #         for cfg in glob.glob(os.path.join(root, "**/adapter_config.json"), recursive=True):
    #             d = os.path.dirname(cfg)
    #             candidates.append((os.path.getmtime(cfg), d))
    #     if not candidates:
    #         raise RuntimeError("No LoRA adapter directories found. Train at least one step first.")
    #     candidates.sort(reverse=True)
    #     return candidates[0][1]
    # =============================================================================
    
    
    # Use iterate_dataset with real training scenarios (similar to train.py)
    training_iterator = iterate_dataset(
        training_scenarios,  # Use real scenarios from Hugging Face
        groups_per_step=training_config["groups_per_step"],
        num_epochs=training_config["num_epochs"],
        initial_step=await model.get_step(),
    )
    
    for batch in training_iterator:
        print(
            f"Training step {batch.step}, epoch {batch.epoch}, epoch step {batch.epoch_step}"
        )
        print(f"Batch contains {len(batch.items)} scenarios")
    
        # Create trajectory groups for this batch (similar to train.py)
        groups = []
        for scenario in batch.items:
            groups.append(
                art.TrajectoryGroup(
                    (
                        wrap_rollout(model, rollout)(
                            model, EmailScenario(step=batch.step, scenario=scenario)
                        )
                        for _ in range(training_config["rollouts_per_group"])
                    )
                )
            )
        print(groups[0])
        # Gather all trajectory groups
        finished_groups = await art.gather_trajectory_groups(
            groups,
            pbar_desc="gather",
            max_exceptions=training_config["rollouts_per_group"] * len(batch.items),
        )
        validate_trajectories(finished_groups) 
    
        judged_groups = []
        for group in finished_groups:
            # Use RULER to assign relative scores to each trajectory
            judged_group = await ruler_score_group(group, "openai/o4-mini", debug=True)
            judged_groups.append(judged_group)
    
        # await model.delete_checkpoints()
        try:
          await model.train(
            judged_groups,
            config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
            # Lowering the logprob_calculation_chunk_size is a memory saving measure
            # to allow longer sequences (up to 8192 tokens) to be processed on a T4.
            _config={"logprob_calculation_chunk_size": 8},
          )
        except httpx.RemoteProtocolError as e:
          print("=" * 80)
          print("[ERROR] RemoteProtocolError during ART training")
          print("可能原因：远端 backend 崩溃 / 超时 / 被杀，请检查 RunPod/SkyPilot 日志")
          print(f"详细异常：{e!r}")
          print("=" * 80)
          raise

# =============================================================================
#     # Save checkpoint to G-Drive
#     try:
#       src = newest_adapter_dir()
#       dst = os.path.join(SAVE_DIR, f"checkpoint_{int(time.time())}")
#       print("Saving adapter from:", src)
#       shutil.copytree(src, dst)
#       print("Saved to:", dst)
#     except Exception as e:
#       print(e)
# 
#     await model.delete_checkpoints()
#     print(f"Completed training step {batch.step}")
# 
#     # Stop after max_steps for demo purposes (adjust as needed)
#     if batch.step >= training_config["max_steps"]:
#         break
# 
# 
#     files_in_directory = os.listdir('.')
#     print("Files in current directory:")
#     for item in files_in_directory:
#       print(item)
#       try:
#         if (item == "gdrive"):
#           continue
#         src = ".art"
#         #dst = os.path.join(SAVE_DIR, f"model_{int(time.time())}")
#         dst = os.path.join(SAVE_DIR, item)
#         #print("Saving adapter from:", src)
#         if (os.path.isdir(item)):
#           shutil.copytree(item, dst)
#         else:
#           shutil.copy(item, dst)
#         print("Saved to:", dst)
#       except Exception as e:
#         print(e)
#     # Save checkpoint to G-Drive
# =============================================================================


# ### Using the Model
# 
# Just like that, you've trained an agent to search emails and answer questions using LangGraph! Now it's time to use your model outside of the training loop.
# 
# Check out the code below for a small demo of the model you just trained!


#@title Loading/inference code

# Test the trained model using the rollout function
# This avoids memory issues and uses the same inference path as training
async def use_model():

    print("Testing the trained LangGraph model with a real scenario...\n")
    
    
    # Use a scenario from our training set
    test_scenario = training_scenarios[1]
    
    print(f"Test scenario ID: {test_scenario.id}")
    print(f"Question: {test_scenario.question}")
    print(f"Expected answer: {test_scenario.answer}")
    print(f"Reference message IDs: {test_scenario.message_ids}")
    print(f"Inbox: {test_scenario.inbox_address}")
    print(f"Query date: {test_scenario.query_date}")
    print("-" * 50)
    
    # Run the rollout function with the trained model
    test_email_scenario = EmailScenario.model_validate(
        {"step": 0, "scenario": test_scenario.model_dump()}
    )
    result_trajectory = await wrap_rollout(model, rollout)(model, test_email_scenario)
    
    print("LangGraph Agent's trajectory:")
    print("-" * 20)
    
    # Display the conversation
    messages = result_trajectory.messages()
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
    
        if role == "system":
            print(
                f"[SYSTEM]: {content[:100]}..."
                if len(content) > 100
                else f"[SYSTEM]: {content}"
            )
        elif role == "user":
            print(f"[USER]: {content}")
        elif role == "assistant":
            if tool_calls:
                print(f"[ASSISTANT]: {tool_calls}")
            if content:
                print(f"[ASSISTANT]: {content}")
        elif role == "tool":
            tool_name = msg.get("name", "unknown_tool")
            print(
                f"[TOOL - {tool_name}]: {content[:200]}..."
                if len(content) > 200
                else f"[TOOL - {tool_name}]: {content}"
            )
    
        print()
    
    print("-" * 50)
    if result_trajectory.final_answer:
        print(f"Agent's Final Answer: {result_trajectory.final_answer.answer}")
        print(f"Source IDs Used: {result_trajectory.final_answer.source_ids}")
    else:
        print("No final answer provided by the agent")
    
    print(f"\nExpected Answer: {test_scenario.answer}")
    print(f"Expected Source IDs: {test_scenario.message_ids}")
    
    print("\n🎉 LangGraph email search agent testing completed!")
    print(
        "The agent used LangGraph's ReAct pattern with the same inference path as during training."
    )

# <div class="align-center">
# <a href="https://github.com/openpipe/art"><img src="https://github.com/openpipe/art/raw/main/assets/ART_pill.png" height="50"></a>
# <a href="https://discord.gg/zbBHRUpwf4"><img src="https://github.com/openpipe/art/raw/main/assets/Discord.png" height="50"></a>
# <a href="https://art.openpipe.ai"><img src="https://github.com/openpipe/art/raw/main/assets/Documentation_pill.png" height="50"></a>
# 
# Questions? Join the Discord and ask away! For feature requests or to leave a star, visit our [Github](https://github.com/openpipe/art).
# 
# </div>

########### Actual execution ##########

# Load training scenarios
training_scenarios = load_training_scenarios(
    split="train", limit=50, max_messages=1, shuffle=True, seed=42
)

print("Email search environment created with full Enron dataset!")
print(
    f"Database contains the complete email dataset, loaded {len(training_scenarios)} training scenarios."
)

# print first scenario
print("\nSample scenario")
print("id:", training_scenarios[0].id)
print("question:", training_scenarios[0].question)
print("answer:", training_scenarios[0].answer)
print("message_ids:", training_scenarios[0].message_ids)
print("how_realistic:", training_scenarios[0].how_realistic)
print("inbox_address:", training_scenarios[0].inbox_address)
print("query_date:", training_scenarios[0].query_date)
print("split:", training_scenarios[0].split)

# print second scenario
print("\nSample scenario")
print("id:", training_scenarios[1].id)
print("question:", training_scenarios[1].question)
print("answer:", training_scenarios[1].answer)
print("message_ids:", training_scenarios[1].message_ids)
print("how_realistic:", training_scenarios[1].how_realistic)
print("inbox_address:", training_scenarios[1].inbox_address)
print("query_date:", training_scenarios[1].query_date)
print("split:", training_scenarios[1].split)


async def main():
    await initialize_model("RTX4090")
    await train()
    await use_model()
    
asyncio.run(main())
