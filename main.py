from dotenv import load_dotenv

from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generation_chain, reflection_chain



load_dotenv()

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]} #The return value is a new state of a dictionary with a messages key
    # The new state (messages) will be autamtically added (appended) to the old state because of the reducer function add_messages

def reflection_node(state: MessageGraph):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content= res.content)]} #we're labeling the AI response as Human Message


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

def should_continue(state: MessageGraph) -> str:
    if len(state["messages"]) > 6:
        return END
    return REFLECT
#The returned value isnt a state so this will NOT be a node, the returned value is a string sand it must corolate to a node name REFLECT, GENERATE

    
builder.add_conditional_edges(GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT}) #Routing function

builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())


def main():
    print("Hello from twitter-post-reflection-agent!")
    inputs = HumanMessage(content= """
        Make this tweet better: "
        @LangChainAI
        - newly Tool Calling feature is seriously underrated.
        After a long wait, it's here - making the implementation of agents across different models with function calling - super easy.
        Made a video covering their newest blog psot.
    """)
    
    response = graph.invoke(inputs)
    print(response["messages"][-1].content)

if __name__ == "__main__":
    main()
