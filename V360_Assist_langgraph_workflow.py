
import pandas as pd
import logging
from langgraph import StateGraph, START, END

# Initialize logging
logging.basicConfig(level=logging.INFO)

def initialize_workflow(topic, start_date, end_date):
    # Validate and initialize state
    return {"topic": topic, "start_date": start_date, "end_date": end_date}

def query_chatbot_conversations(state):
    # Query logic here
    return bot_df

def query_call_summaries(state):
    # Query logic here
    return summary_df

def count_tokens(text):
    # Token counting logic
    return token_count

def batch_chatbot_data(bot_df):
    # Batching logic
    return bot_df_batches

def batch_call_summaries(summary_df):
    # Batching logic
    return summary_df_batches

def vbot_analysis(batch):
    # LLM call logic
    return vbot_issues

def call_summary_analysis(batch):
    # LLM call logic
    return call_points

def generate_insights(vbot_issues, call_points):
    # Synthesize insights into markdown
    return final_report

# Define the graph
builder = StateGraph()
builder.add_node(initialize_workflow)
builder.add_node(query_chatbot_conversations)
builder.add_node(query_call_summaries)
builder.add_node(batch_chatbot_data)
builder.add_node(batch_call_summaries)
builder.add_node(vbot_analysis)
builder.add_node(call_summary_analysis)
builder.add_node(generate_insights)

# Connect nodes with edges
builder.add_edge(START, "initialize_workflow")
builder.add_edge("initialize_workflow", "query_chatbot_conversations")
builder.add_edge("initialize_workflow", "query_call_summaries")
builder.add_edge("query_chatbot_conversations", "batch_chatbot_data")
builder.add_edge("query_call_summaries", "batch_call_summaries")
builder.add_edge("batch_chatbot_data", "vbot_analysis")
builder.add_edge("batch_call_summaries", "call_summary_analysis")
builder.add_edge("vbot_analysis", "generate_insights")
builder.add_edge("call_summary_analysis", "generate_insights")
builder.add_edge("generate_insights", END)

# Compile and run the graph
graph = builder.compile()
graph.invoke(initial_state)


#################################################################################################################################################

from datetime import datetime

def initialize_workflow(topic: str, start_date: str, end_date: str) -> dict:
    """
    Initializes the workflow state with the provided parameters.

    Args:
        topic (str): The focus area for the analysis (e.g., "billing issues").
        start_date (str): The start date for the analysis in 'YYYY-MM-DD' format.
        end_date (str): The end date for the analysis in 'YYYY-MM-DD' format.

    Returns:
        dict: A dictionary containing the initialized parameters.

    Raises:
        ValueError: If the date format is incorrect or if the start date is after the end date.
    """
    # Validate the date format
    try:
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        raise ValueError("Date format should be 'YYYY-MM-DD'.") from e

    # Check if the start date is after the end date
    if start_date_obj > end_date_obj:
        raise ValueError("Start date cannot be after end date.")

    # Initialize the state dictionary
    state = {
        "topic": topic,
        "start_date": start_date,
        "end_date": end_date
    }

    return state


import pandas as pd

def query_chatbot_conversations(state: dict) -> pd.DataFrame:
    """
    Queries the chatbot conversation table based on the provided state parameters.

    Args:
        state (dict): A dictionary containing the initialized parameters (topic, start_date, end_date).

    Returns:
        pd.DataFrame: A DataFrame containing the chatbot transcripts that match the criteria.
    """
    # Extract parameters from the state
    topic = state['topic']
    start_date = state['start_date']
    end_date = state['end_date']

    # Placeholder SQL query (to be replaced with actual query logic)
    query = f"""
    SELECT transcripts
    FROM chatbot_conversations
    WHERE topic = '{topic}' AND conversation_date BETWEEN '{start_date}' AND '{end_date}'
    """

    # Execute the query and return the results as a DataFrame
    # Placeholder for actual database execution
    # Replace 'spart.sql(query).to_pandas()' with the actual database call
    bot_df = pd.DataFrame()  # Placeholder for the actual DataFrame returned from the query
    return bot_df


def query_call_summaries(state: dict) -> pd.DataFrame:
    """
    Queries the call summary table based on the provided state parameters.

    Args:
        state (dict): A dictionary containing the initialized parameters (topic, start_date, end_date).

    Returns:
        pd.DataFrame: A DataFrame containing the call summaries that match the criteria.
    """
    # Extract parameters from the state
    topic = state['topic']
    start_date = state['start_date']
    end_date = state['end_date']

    # Placeholder SQL query (to be replaced with actual query logic)
    query = f"""
    SELECT summary
    FROM call_summaries
    WHERE topic = '{topic}' AND call_date BETWEEN '{start_date}' AND '{end_date}'
    """

    # Execute the query and return the results as a DataFrame
    # Placeholder for actual database execution
    # Replace 'spart.sql(query).to_pandas()' with the actual database call
    summary_df = pd.DataFrame()  # Placeholder for the actual DataFrame returned from the query
    return summary_df

import tiktoken

def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in the provided text using the tiktoken library.

    Args:
        text (str): The text for which to count tokens.

    Returns:
        int: The number of tokens in the text.
    """
    # Initialize the tokenizer
    encoder = tiktoken.get_encoding("cl100k_base")  # Using the encoding for OpenAI models

    # Encode the text to get the token representation
    tokenized_text = encoder.encode(text)

    # Return the number of tokens
    return len(tokenized_text)

import pandas as pd

def batch_chatbot_data(bot_df: pd.DataFrame) -> list:
    """
    Splits the chatbot DataFrame into manageable batches based on token count.

    Args:
        bot_df (pd.DataFrame): A DataFrame containing chatbot transcripts.

    Returns:
        list: A list of DataFrames, each representing a batch of chatbot data.
    """
    # Initialize a list to hold the batches
    bot_df_batches = []

    # Concatenate all transcripts into a single string for token counting
    all_transcripts = " ".join(bot_df['transcripts'].tolist())

    # Count the total number of tokens in the concatenated transcripts
    total_tokens = count_tokens(all_transcripts)

    # Determine the number of batches based on the total token count
    if total_tokens <= 100000:
        # If total tokens are less than or equal to 100,000, create a single batch
        bot_df_batches.append(bot_df)
    else:
        # Calculate the number of batches needed
        num_batches = total_tokens // 80000 + 1  # Each batch can have up to 80,000 tokens

        # Split the DataFrame into batches
        batch_size = len(bot_df) // num_batches
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size if i < num_batches - 1 else len(bot_df)
            bot_df_batches.append(bot_df.iloc[start_index:end_index])

    return bot_df_batches

def batch_call_summaries(summary_df: pd.DataFrame) -> list:
    """
    Splits the call summary DataFrame into manageable batches based on token count.

    Args:
        summary_df (pd.DataFrame): A DataFrame containing call summaries.

    Returns:
        list: A list of DataFrames, each representing a batch of call summary data.
    """
    # Initialize a list to hold the batches
    summary_df_batches = []

    # Concatenate all summaries into a single string for token counting
    all_summaries = " ".join(summary_df['summary'].tolist())

    # Count the total number of tokens in the concatenated summaries
    total_tokens = count_tokens(all_summaries)

    # Determine the number of batches based on the total token count
    if total_tokens <= 100000:
        # If total tokens are less than or equal to 100,000, create a single batch
        summary_df_batches.append(summary_df)
    else:
        # Calculate the number of batches needed
        num_batches = total_tokens // 80000 + 1  # Each batch can have up to 80,000 tokens

        # Split the DataFrame into batches
        batch_size = len(summary_df) // num_batches
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size if i < num_batches - 1 else len(summary_df)
            summary_df_batches.append(summary_df.iloc[start_index:end_index])

    return summary_df_batches
    
    

from typing import TypedDict, List, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
import openai

class VBOTAnalysisState(TypedDict):
    """
    State for VBOT Analysis Workflow
    """
    bot_df_batches: List[str]  # Input batches of chatbot transcripts
    identified_issues: Annotated[List[List[str]], operator.add]  # Accumulate issues from each batch
    final_routing_issues: List[str]  # Consolidated unique routing issues

def vbot_analysis_map(batch: str) -> dict:
    """
    Analyze a single batch of chatbot transcripts to identify routing issues.

    Args:
        batch (str): A single batch of chatbot transcripts.

    Returns:
        dict: A dictionary containing identified routing issues for the batch.
    """
    try:
        # Construct a detailed prompt for routing issue analysis
        prompt = f"""
        Analyze the following chatbot transcripts and identify specific routing issues:

        Transcripts:
        {batch}

        Routing Issue Analysis Instructions:
        1. Identify clear routing problems or inconsistencies
        2. Categorize each issue (e.g., technical routing, support escalation)
        3. Provide a brief context for each identified issue
        4. Be concise and specific

        Output Format:
        - Issue Category: [Category]
        - Description: [Specific Routing Issue]
        """

        # Azure OpenAI API call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are an expert analyst identifying routing issues in chatbot transcripts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )

        # Extract routing issues from the response
        routing_issues = response['choices'][[0]](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/)['message']['content'].strip().split('\n')
        
        return {"identified_issues": [routing_issues]}

    except Exception as e:
        print(f"Error in vbot_analysis_map: {e}")
        return {"identified_issues": []}


def map_vbot_batches(state: VBOTAnalysisState) -> List[Send]:
    """
    Distribute batches for parallel processing using Send API.

    Args:
        state (VBOTAnalysisState): Current workflow state.

    Returns:
        List[Send]: Send actions for parallel batch processing.
    """
    return [
        Send("vbot_analysis_node", {"batch": batch}) 
        for batch in state['bot_df_batches']
    ]


def vbot_analysis_reduce(state: VBOTAnalysisState) -> dict:
    """
    Consolidate routing issues from all batches.

    Args:
        state (VBOTAnalysisState): Current workflow state.

    Returns:
        dict: Consolidated unique routing issues.
    """
    # Flatten and deduplicate routing issues
    all_issues = set()
    for batch_issues in state['identified_issues']:
        all_issues.update(batch_issues)
    
    return {
        "final_routing_issues": list(all_issues)
    }


def create_vbot_analysis_workflow() -> StateGraph:
    """
    Create a LangGraph workflow for VBOT transcript analysis.

    Returns:
        StateGraph: Configured workflow for parallel batch processing.
    """
    # Create the workflow graph
    workflow = StateGraph(VBOTAnalysisState)

    # Add nodes
    workflow.add_node(START, map_vbot_batches)
    workflow.add_
