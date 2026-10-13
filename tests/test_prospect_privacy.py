import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ["LANGSMITH_TRACING"] = "false"
agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-TEST"
        self.billing = {
            "tax_id": "123-45-6789",
            "date_of_birth": "1980-01-02",
            "card_on_file": "4111111111111111",
            "credit_check_ref": "REF-123",
        }
        self.sensitive_keys = tuple(self.billing)
        self.record = {
            "name": "Ada Prospect",
            "email": "ada@example.com",
            "disqualified": False,
            "billing_qualification": self.billing,
            "annual_revenue": 90_000_000,
            "enrichment_source": "test-provider",
        }

    def test_get_prospect_returns_only_allowlisted_contact_fields(self):
        with patch.object(agent.data_service, "get_prospect_record", return_value=self.record):
            result = agent.get_prospect.func(self.prospect_id)

        self.assertEqual(result["prospect"], {
            "prospect_id": self.prospect_id,
            "name": "Ada Prospect",
            "email": "ada@example.com",
            "disqualified": False,
        })
        self.assertNotIn("billing_qualification", result["prospect"])
        serialized = json.dumps(result)
        for key in self.sensitive_keys:
            self.assertNotIn(key, serialized)

    def test_allowlisted_prospect_can_still_be_emailed(self):
        runtime = SimpleNamespace(config={"metadata": {"user_id": "REP-TEST"}})
        with (
            patch.object(agent.data_service, "get_prospect_record", return_value=self.record),
            patch.object(agent.data_service, "get_rep", return_value={"name": "Rep", "email": "rep@example.com"}),
        ):
            prospect = agent.get_prospect.func(self.prospect_id)["prospect"]
            result = agent.send_prospect_email.func(
                prospect, "Hello", "Checking in", runtime
            )

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "ada@example.com")
        self.assertNotIn("billing_qualification", result)

    def test_build_prospect_profile_omits_billing_and_contact_pii(self):
        with (
            patch.object(agent.data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(agent.data_service, "get_prospect_record", return_value=self.record),
            patch.object(agent.data_service, "fetch_engagement_history", return_value=[]),
            patch.object(agent.data_service, "fetch_account_details", return_value=[{"industry": "Software"}]),
            patch.object(agent.data_service, "fetch_tech_stack", return_value=["Kafka"]),
            patch.object(agent.data_service, "save_profile_to_db") as save_profile,
        ):
            result = agent.build_prospect_profile.func(self.prospect_id)

        profile = result["prospect_profile"]
        self.assertEqual(profile["annual_revenue"], 90_000_000)
        self.assertEqual(profile["tech_stack"], ["Kafka"])
        self.assertNotIn("billing_qualification", profile)
        self.assertNotIn("name", profile)
        self.assertNotIn("email", profile)
        serialized = json.dumps(save_profile.call_args.args[1])
        for key in self.sensitive_keys:
            self.assertNotIn(key, serialized)

    def test_build_prospect_profile_sanitizes_existing_cached_profile(self):
        cached = {
            "prospect_id": self.prospect_id,
            "name": "Ada Prospect",
            "email": "ada@example.com",
            "annual_revenue": 90_000_000,
            "billing_qualification": self.billing,
            "tech_stack": ["Kafka"],
            "account_details": [{"industry": "Software"}],
        }
        with (
            patch.object(agent.data_service, "get_profile_from_db", return_value={"prospect_profile": cached}),
            patch.object(agent.data_service, "save_profile_to_db") as save_profile,
        ):
            result = agent.build_prospect_profile.func(self.prospect_id)

        profile = result["prospect_profile"]
        self.assertEqual(profile["annual_revenue"], 90_000_000)
        self.assertEqual(profile["tech_stack"], ["Kafka"])
        self.assertNotIn("billing_qualification", profile)
        self.assertNotIn("email", profile)
        serialized = json.dumps(save_profile.call_args.args[1])
        for key in self.sensitive_keys:
            self.assertNotIn(key, serialized)

    def test_score_prospect_filters_profile_before_nested_model_input(self):
        captured = {}

        def invoke(messages):
            captured["messages"] = messages
            return SimpleNamespace(model_dump=lambda: {"score": 85})

        profile = {
            "prospect_id": self.prospect_id,
            "name": "Ada Prospect",
            "email": "ada@example.com",
            "annual_revenue": 90_000_000,
            "account_details": [{"industry": "Software"}],
            "tech_stack": ["Kafka"],
            "billing_qualification": self.billing,
        }
        offering = {
            "required_tech_stack": ["Kafka"],
            "min_annual_revenue": 50_000_000,
            "description": "Test offering",
        }
        with (
            patch.object(agent.data_service, "fetch_tech_stack", return_value=["Kafka"]),
            patch.object(agent, "_scoring_llm", SimpleNamespace(invoke=invoke)),
        ):
            result = agent.score_prospect.func(profile, offering)

        user = captured["messages"][1]["content"]
        serialized_profile = json.loads(user.split("\n\nProspect profile:\n", 1)[1])
        self.assertEqual(result["score"], 85)
        self.assertEqual(serialized_profile["annual_revenue"], 90_000_000)
        self.assertEqual(serialized_profile["tech_stack"], ["Kafka"])
        self.assertEqual(serialized_profile["account_details"], [{"industry": "Software"}])
        self.assertNotIn("billing_qualification", serialized_profile)
        self.assertNotIn("email", serialized_profile)
        for key in self.sensitive_keys:
            self.assertNotIn(key, user)

    def test_data_service_sanitizes_cached_profiles(self):
        from gtm_agent import data_service

        data_service._PROFILES[self.prospect_id] = {
            "prospect_id": self.prospect_id,
            "annual_revenue": 90_000_000,
            "billing_qualification": self.billing,
            "tech_stack": ["Kafka"],
        }
        try:
            result = data_service.get_profile_from_db(self.prospect_id)
        finally:
            data_service._PROFILES.pop(self.prospect_id, None)

        self.assertEqual(result["prospect_profile"]["tech_stack"], ["Kafka"])
        self.assertNotIn("billing_qualification", result["prospect_profile"])

        data_service.save_profile_to_db(self.prospect_id, {
            "prospect_id": self.prospect_id,
            "annual_revenue": 90_000_000,
            "billing_qualification": self.billing,
            "tech_stack": ["Kafka"],
        })
        try:
            saved_profile = data_service._PROFILES[self.prospect_id]
        finally:
            data_service._PROFILES.pop(self.prospect_id, None)

        self.assertNotIn("billing_qualification", saved_profile)
        for key in self.sensitive_keys:
            self.assertNotIn(key, json.dumps(saved_profile))


if __name__ == "__main__":
    unittest.main()
