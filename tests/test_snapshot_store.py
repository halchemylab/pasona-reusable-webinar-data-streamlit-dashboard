import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import snapshot_store


class TestSnapshotStore(unittest.TestCase):
    def test_snapshot_roundtrip(self):
        with TemporaryDirectory() as tmp:
            original_history_file = snapshot_store.HISTORY_FILE
            snapshot_store.HISTORY_FILE = Path(tmp) / "webinar_history.csv"
            try:
                emails_df = pd.DataFrame(
                    [
                        {
                            "name": "Email A",
                            "open_rate": 40.0,
                            "unique_ctr": 9.0,
                            "total_delivered": 100,
                            "unique_opens": 40,
                            "unique_clicks": 9,
                        }
                    ]
                )
                landing = {"views": 250, "active_users": 180}
                social = {"linkedin": {"impressions": 1200, "engagements": 60, "clicks": 15}, "facebook": {"views": 300}}
                regs_df = pd.DataFrame([{"name": "Alice", "company": "Acme"}])
                survey = {"n_responses": 10, "consult_yes_count": 2, "consult_no_count": 8, "top_themes": []}
                summary = "Strong campaign performance."

                row = snapshot_store.build_snapshot_row("Webinar X", emails_df, landing, social, regs_df, survey, summary)
                snapshot_store.append_snapshot_row(row)

                hist = snapshot_store.load_snapshot_history()
                self.assertEqual(len(hist), 1)
                webinar_id = str(hist.iloc[0]["webinar_id"])
                state = snapshot_store.load_snapshot_into_state(webinar_id)
                self.assertIn("parsed_emails_df", state)
                self.assertIn("landing_metrics_dict", state)
                self.assertEqual(state["landing_metrics_dict"].get("views"), 250)
                self.assertEqual(len(state["parsed_emails_df"]), 1)
                self.assertEqual(state["exec_summary_text"], summary)
            finally:
                snapshot_store.HISTORY_FILE = original_history_file

    def test_has_snapshot_data(self):
        self.assertFalse(
            snapshot_store.has_snapshot_data(
                pd.DataFrame(),
                {},
                {},
                pd.DataFrame(),
                {},
                "",
            )
        )
        self.assertTrue(
            snapshot_store.has_snapshot_data(
                pd.DataFrame([{"name": "x"}]),
                {},
                {},
                pd.DataFrame(),
                {},
                "",
            )
        )


if __name__ == "__main__":
    unittest.main()
