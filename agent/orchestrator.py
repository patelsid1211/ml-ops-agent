import json
import logging
import os
from dotenv import load_dotenv
import anthropic

from agent.tools import query_feature_store, run_evaluation, trigger_retrain, send_alert

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL  = "claude-opus-4-6"

# --- System prompt ---
SYSTEM_PROMPT = """
You are an ML Ops monitoring agent responsible for maintaining the health of a 
CO air pollution prediction model in production.

Your job is to:
1. Check the current model performance by running an evaluation
2. Query the feature store to understand the data
3. Based on the results, decide the appropriate action:
   - If the model is healthy (RMSE < 0.5 and R2 > 0.85): report the status, no action needed
   - If metrics are borderline (alert status): send an alert to notify the team
   - If performance has degraded (retrain status): trigger retraining and then send a summary alert
4. Always explain your reasoning clearly before taking any action

Be concise, precise, and always justify your decisions with the actual metric values.
"""

# --- Tool definitions sent to the Anthropic API ---
TOOLS = [
    {
        "name": "query_feature_store",
        "description": "Query the feature store to get statistics about the latest data version including row count, feature count, and target distribution.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "run_evaluation",
        "description": "Run the evaluation harness against the latest trained model. Returns metrics (RMSE, MAE, R2) and a status: healthy, alert, or retrain.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "trigger_retrain",
        "description": "Trigger a full retraining pipeline using the latest feature version. Call this when evaluation status is retrain.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "send_alert",
        "description": "Send an alert notification with a message and severity level.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The alert message to send"
                },
                "level": {
                    "type": "string",
                    "enum": ["info", "warning", "critical"],
                    "description": "Severity level of the alert"
                }
            },
            "required": ["message", "level"]
        }
    }
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute the tool the agent requested and return result as a string."""
    logger.info(f"Executing tool: {tool_name} with input: {tool_input}")

    if tool_name == "query_feature_store":
        result = query_feature_store()
    elif tool_name == "run_evaluation":
        result = run_evaluation()
    elif tool_name == "trigger_retrain":
        result = trigger_retrain()
    elif tool_name == "send_alert":
        result = send_alert(
            message=tool_input.get("message", ""),
            level=tool_input.get("level", "warning")
        )
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result, default=str)


def run_agent(user_message: str = None) -> str:
    """Run the agentic loop until the agent completes its task.
    
    The loop:
    1. Send message to agent with available tools
    2. If agent wants to call a tool → execute it → send result back
    3. Repeat until agent returns a final text response
    """
    if user_message is None:
        user_message = "Please monitor the ML pipeline. Check model performance and take appropriate action."

    messages = [{"role": "user", "content": user_message}]
    logger.info("Agent started")

    # Agentic loop
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        logger.info(f"Agent stop reason: {response.stop_reason}")

        # Add agent response to message history
        messages.append({"role": "assistant", "content": response.content})

        # If agent is done thinking and has no tool calls → return final response
        if response.stop_reason == "end_turn":
            final_text = next(
                (block.text for block in response.content if hasattr(block, "text")), ""
            )
            logger.info("Agent completed task")
            return final_text

        # If agent wants to call tools → execute them
        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type"       : "tool_result",
                        "tool_use_id": block.id,
                        "content"    : tool_result
                    })

            # Send tool results back to agent
            messages.append({"role": "user", "content": tool_results})


def run_agent_mock() -> str:
    """Mock agent run for testing without Anthropic API credits.
    
    Simulates the full agentic loop:
    1. Query feature store
    2. Run evaluation
    3. Decide action based on status
    4. Send alert or trigger retrain if needed
    """
    print("\n[MOCK MODE] Simulating agent decision loop...\n")

    # Step 1: query feature store
    print(">> Tool call: query_feature_store")
    store_stats = query_feature_store()
    print(f"   Result: {json.dumps(store_stats, indent=2)}\n")

    # Step 2: run evaluation
    print(">> Tool call: run_evaluation")
    report = run_evaluation()
    print(f"   Status : {report['status']}")
    print(f"   RMSE   : {report['metrics']['rmse']:.4f}")
    print(f"   MAE    : {report['metrics']['mae']:.4f}")
    print(f"   R2     : {report['metrics']['r2']:.4f}\n")

    # Step 3: decide action based on status
    status = report["status"]

    if status == "healthy":
        msg = (f"Model is healthy. RMSE={report['metrics']['rmse']:.4f}, "
               f"R2={report['metrics']['r2']:.4f}. No action needed.")
        print(">> Agent decision: model is healthy, no action needed")
        print(">> Tool call: send_alert (info)")
        send_alert(message=msg, level="info")

    elif status == "alert":
        msg = (f"Model performance is borderline. RMSE={report['metrics']['rmse']:.4f}, "
               f"R2={report['metrics']['r2']:.4f}. Human review recommended.")
        print(">> Agent decision: borderline performance, sending alert")
        print(">> Tool call: send_alert (warning)")
        send_alert(message=msg, level="warning")

    elif status == "retrain":
        msg = (f"Model performance degraded. RMSE={report['metrics']['rmse']:.4f}, "
               f"R2={report['metrics']['r2']:.4f}. Triggering retraining.")
        print(">> Agent decision: performance degraded, triggering retrain")
        print(">> Tool call: trigger_retrain")
        retrain_result = trigger_retrain()
        print(f"   Result: {retrain_result}\n")
        print(">> Tool call: send_alert (critical)")
        send_alert(message=msg, level="critical")

    summary = (f"Agent completed monitoring cycle. "
               f"Status: {status}. "
               f"Data version: {store_stats['latest_version']}. "
               f"Rows: {store_stats['num_rows']}.")
    return summary


if __name__ == "__main__":
    import sys
    mock = "--mock" in sys.argv or True  # default to mock until credits are added
    if mock:
        result = run_agent_mock()
    else:
        result = run_agent()
    print("\nAgent Final Report:")
    print("-" * 50)
    print(result)