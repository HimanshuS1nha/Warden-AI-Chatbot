from langchain_core.messages import AIMessage, HumanMessage

from warden.graph.graph import DEMO_REPLY, build_graph


async def test_demo_graph_appends_a_reply() -> None:
    graph = build_graph()

    result = await graph.ainvoke({"messages": [HumanMessage(content="Hi")]})

    question, reply = result["messages"]
    assert question.content == "Hi"
    assert isinstance(reply, AIMessage)
    assert reply.content == DEMO_REPLY
