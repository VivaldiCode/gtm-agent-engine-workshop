import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from gtm_agent import gtm_agent as module


class RepContextTests(unittest.TestCase):
    @patch.object(module.gtm_agent, "invoke")
    @patch.object(module.data_service, "get_rep", return_value={"name": "Avery", "email": "avery@example.com"})
    def test_resolved_rep_is_invocation_context(self, get_rep, invoke):
        invoke.return_value = {"messages": [SimpleNamespace(content="done")]}

        module.run_agent("Draft an email", user_id="rep-1")

        messages = invoke.call_args.args[0]["messages"]
        self.assertEqual(messages[0], {
            "role": "system",
            "content": "Signed-in rep context: Name: Avery; Email: avery@example.com",
        })
        self.assertEqual(messages[1], {"role": "user", "content": "Draft an email"})
        get_rep.assert_called_once_with("rep-1")
        self.assertEqual(invoke.call_args.kwargs["config"]["metadata"]["user_id"], "rep-1")

    @patch.object(module.gtm_agent, "invoke")
    @patch.object(module.data_service, "get_rep", side_effect=RuntimeError("directory unavailable"))
    def test_failed_rep_lookup_preserves_request_without_identity(self, get_rep, invoke):
        invoke.return_value = {"messages": [SimpleNamespace(content="done")]}

        module.run_agent("Email this prospect", user_id="rep-1")

        self.assertEqual(invoke.call_args.args[0]["messages"], [
            {"role": "user", "content": "Email this prospect"},
        ])
        self.assertNotIn("first step", module.SYSTEM_PROMPT)
        tools = module.gtm_agent.get_graph().nodes["tools"].data._tools_by_name
        self.assertIn("get_current_rep", tools)
        get_rep.assert_called_once_with("rep-1")

    @patch.object(module.data_service, "get_rep", return_value={"name": "Avery", "email": "avery@example.com"})
    def test_email_tool_keeps_signed_in_sender_default(self, get_rep):
        runtime = SimpleNamespace(config={"metadata": {"user_id": "rep-1"}})

        result = module.send_prospect_email.func(
            prospect={"name": "Jordan", "email": "jordan@example.com"},
            subject="Hello",
            body="Hi Jordan",
            runtime=runtime,
        )

        self.assertEqual(result["from"], "avery@example.com")
        self.assertEqual(result["from_name"], "Avery")
        get_rep.assert_called_once_with("rep-1")

    @patch.object(module.data_service, "get_rep", return_value={"name": "Avery", "email": "avery@example.com"})
    @patch.object(
        module.agent_model,
        "_generate",
        side_effect=[
            ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="",
                tool_calls=[{
                    "name": "send_prospect_email",
                    "args": {
                        "prospect": {"name": "Jordan", "email": "jordan@example.com"},
                        "subject": "Hello",
                        "body": "Hi Jordan,\n\nBest, Avery",
                    },
                    "id": "call-email",
                    "type": "tool_call",
                }],
            ))]),
            ChatResult(generations=[ChatGeneration(message=AIMessage(content="Email sent."))]),
        ],
    )
    def test_email_flow_uses_rep_context_without_rep_tool_round_trip(self, generate, get_rep):
        result = module.run_agent("Email Jordan", user_id="rep-1")

        self.assertEqual(generate.call_count, 2)
        first_prompt = generate.call_args_list[0].args[0]
        self.assertTrue(any(
            message.type == "system"
            and "Name: Avery; Email: avery@example.com" in str(message.content)
            for message in first_prompt
        ))
        self.assertEqual(
            [message.name for message in result["messages"] if getattr(message, "name", None)],
            ["send_prospect_email"],
        )
        self.assertEqual(result["reply"], "Email sent.")
        self.assertEqual(get_rep.call_count, 2)
        get_rep.assert_any_call("rep-1")

    @patch.object(module.data_service, "get_rep", return_value={"name": "Avery", "email": "avery@example.com"})
    @patch.object(
        module.agent_model,
        "_generate",
        return_value=ChatResult(generations=[ChatGeneration(message=AIMessage(content="Score complete."))]),
    )
    def test_scoring_flow_completes_in_one_model_call(self, generate, get_rep):
        result = module.run_agent("Score this prospect", user_id="rep-1")

        self.assertEqual(generate.call_count, 1)
        self.assertEqual(result["reply"], "Score complete.")
        get_rep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
