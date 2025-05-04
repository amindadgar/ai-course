"""
LangChain examples similar to llama-index and OpenAI examples
"""
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import pandas as pd

# Load environment variables
load_dotenv()

print("=== Simple Query ===")
# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o-mini")

# Create a list of messages
messages = [
    SystemMessage(content="You are an AI assistant"),
    HumanMessage(content="Who are you?")
]

# Get a response
response = llm.invoke(messages)
print(f"Response: {response.content}")

print("\n=== Temperature Experiment ===")
# List of different temperature values to test
temperatures = [0.0, 0.3, 0.7, 1.0, 1.5]
results = []

# The prompt to use for all temperature tests
prompt = "Write a very short explanation for artificial intelligence (max 2 sentences)"

# System message to use
system_message = "You are a helpful AI assistant"

for temp in temperatures:
    # Initialize the LLM with the current temperature
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temp)
    
    print(f"Temperature: {temp}")
    for i in range(3):
        # Create message list
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ]
        
        # Get a response
        response = llm.invoke(messages)
        
        # Save the results
        results.append({
            "Temperature": temp,
            "Response": response.content
        })
    
        print(f"{response.content}")
    print("-" * 50)

# Create a DataFrame to display the results more clearly
df = pd.DataFrame(results)
df.to_csv("langchain_temperature_results.csv", index=False)

print("\n=== Chat Memory Example ===")
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Create a conversation chain with memory
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    memory=memory,
    verbose=False
)

# Start an interactive chat session
response = conversation.invoke({"input": "Hello, how are you?"})
print(f"AI: {response['response']}")

response = conversation.invoke({"input": "What is the capital of France?"})
print(f"AI: {response['response']}")

response = conversation.invoke({"input": "What was our conversation about?"})
print(f"AI: {response['response']}")

print("\n=== Reset Memory ===")
# Reset the memory
memory.clear()

response = conversation.invoke({"input": "What was our conversation about?"})
print(f"AI: {response['response']}")

print("\n=== Tool Usage Example ===")
from langchain.agents import tool
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain_openai import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define a tool
@tool
def get_weather(location: str) -> str:
    """Get the weather in a specific location"""
    # This would typically call a weather API
    return f"The weather in {location} is sunny and 75 degrees Fahrenheit."

# Define the tools
tools = [get_weather]

# Get the OpenAI functions
functions = [format_tool_to_openai_function(tool) for tool in tools]

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind(functions=functions)

# Create the agent
agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x.get("chat_history", []),
        "agent_scratchpad": lambda x: format_log_to_str(x.get("intermediate_steps", [])),
    }
    | prompt
    | llm_with_tools
    | ReActSingleInputOutputParser()
)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
result = agent_executor.invoke({"input": "What's the weather like in New York?"})
print(f"Final Result: {result['output']}") 