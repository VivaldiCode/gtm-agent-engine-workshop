import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "sk-test")

from langchain_core.messages import AIMessage

from gtm_agent import gtm_agent as agent_module
from gtm_agent.gtm_records import PROSPECTS


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_mrossi"}})

    def test_disqualified_prospect_is_blocked_before_message_id(self):
        with patch.object(agent_module.uuid, "uuid4") as new_uuid:
            result = agent_module.send_prospect_email.func(
                prospect={"prospect_id": "LEAD-50001", "email": "untrusted@example.com"},
                subject="Demo",
                body="Invite",
                runtime=self.runtime,
            )

        self.assertEqual(
            result,
            {"status": "blocked", "error": "Prospect is disqualified; outreach is not permitted."},
        )
        new_uuid.assert_not_called()

    def test_unknown_prospect_is_not_sent(self):
        with (
            patch.object(agent_module.data_service, "get_prospect_record", return_value=None),
            patch.object(agent_module.uuid, "uuid4") as new_uuid,
        ):
            result = agent_module.send_prospect_email.func(
                prospect={"prospect_id": "LEAD-UNKNOWN"},
                subject="Demo",
                body="Invite",
                runtime=self.runtime,
            )

        self.assertEqual(result, {"status": "failed", "error": "Unknown prospect."})
        new_uuid.assert_not_called()

    def test_eligible_prospect_uses_authoritative_email_and_sends(self):
        record = PROSPECTS["LEAD-12853"]
        with patch.object(agent_module.uuid, "uuid4", return_value=SimpleNamespace(hex="123456789abc")):
            result = agent_module.send_prospect_email.func(
                prospect={
                    "prospect_id": "LEAD-12853",
                    "name": "Untrusted Name",
                    "email": "untrusted@example.com",
                },
                subject="Demo",
                body="Invite",
                runtime=self.runtime,
            )

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], record["email"])
        self.assertEqual(result["to_name"], record["name"])
        self.assertEqual(result["message_id"], "msg-123456789abc")


class DisqualifiedEmailRequestTests(unittest.TestCase):
    def test_lead_50001_request_is_blocked_and_explained(self):
        self.assertIn("Never email a prospect whose record is disqualified.", agent_module.SYSTEM_PROMPT)
        turns = iter((
            AIMessage(content="", tool_calls=[{
                "name": "get_current_rep",
                "args": {},
                "id": "call_rep",
                "type": "tool_call",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "get_prospect",
                "args": {"prospect_id": "LEAD-50001"},
                "id": "call_prospect",
                "type": "tool_call",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "send_prospect_email",
                "args": {
                    "prospect": {"prospect_id": "LEAD-50001", "email": "untrusted@example.com"},
                    "subject": "Demo invitation",
                    "body": "Please book a demo.",
                },
                "id": "call_email",
                "type": "tool_call",
            }]),
        ))

        def fake_invoke(model, messages, config=None, **kwargs):
            try:
                return next(turns)
            except StopIteration:
                self.assertIn(
                    "Prospect is disqualified; outreach is not permitted.",
                    str(messages[-1].content),
                )
                return AIMessage(content="The email was not sent because the prospect is disqualified.")

        with (
            patch.object(agent_module.agent_model.__class__, "invoke", fake_invoke),
            patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}),
        ):
            result = agent_module.run_agent(
                "Email LEAD-50001 to invite them to book a demo.",
                user_id="rep_mrossi",
            )

        tool_results = [message.content for message in result["messages"] if getattr(message, "name", None) == "send_prospect_email"]
        self.assertTrue(any("blocked" in str(content) for content in tool_results))
        self.assertIn("not sent", result["reply"])
        self.assertIn("disqualified", result["reply"])


if __name__ == "__main__":
    unittest.main()
