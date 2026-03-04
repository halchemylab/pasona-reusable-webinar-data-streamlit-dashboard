import unittest

from parsers import parse_emails, parse_regs


class TestParsers(unittest.TestCase):
    def test_parse_emails_rule_based_two_blocks(self):
        text = """
Click-Through Rate Report
Name: Campaign A
Subject Promo A Tracker Domain
Started At 2026-02-10 10:00 Created At 2026-02-09 09:00
Total Sent 1,000
Total Delivered 980
Total Failed 20
Delivery Rate 98.0%
Unique HTML Opens 420
HTML Open Rate 42.9%
Unique Clicks 90
Unique Click Through Rate 9.2%
Click to Open Ratio 21.4%
Total Opt Outs 3
Page 1 of 1

Click-Through Rate Report
Name: Campaign B
Subject Promo B Tracker Domain
Started At 2026-02-11 10:00 Created At 2026-02-10 09:00
Total Sent 500
Total Delivered 490
Total Failed 10
Delivery Rate 98.0%
Unique HTML Opens 200
HTML Open Rate 40.8%
Unique Clicks 45
Unique Click Through Rate 9.2%
Click to Open Ratio 22.5%
Total Opt Outs 2
Page 1 of 1
""".strip()
        df, dbg, ok = parse_emails(text, api_key="", model="gpt-5-mini", temp=0.2)
        self.assertTrue(ok, msg=f"debug={dbg}")
        self.assertEqual(len(df), 2)
        self.assertEqual(int(df["total_delivered"].sum()), 1470)
        self.assertAlmostEqual(float(df["open_rate"].iloc[0]), 42.9, places=1)

    def test_parse_regs_rule_based_rows(self):
        text = """
Name: Alice Tan | Company: Acme Inc | Score: 10 | Last Submitted: 2026-02-11 10:00 | Last Activity: 2026-02-11 10:05
Name: Bob Lee | Company: Beta LLC | Score: 7 | Last Submitted: 2026-02-12 11:00 | Last Activity: 2026-02-12 11:05
""".strip()
        df, dbg, ok = parse_regs(text, api_key="", model="gpt-5-mini", temp=0.2)
        self.assertTrue(ok, msg=dbg)
        self.assertFalse(df.empty)
        self.assertIn("name", df.columns)
        self.assertIn("company", df.columns)
        self.assertEqual(set(df["name"].tolist()), {"Alice Tan", "Bob Lee"})

    def test_parse_regs_list_summary_fallback(self):
        text = """
webinar_demo_registrants
120
Total prospects
95
Mailable prospects
79.2%
Mailable
""".strip()
        df, dbg, ok = parse_regs(text, api_key="", model="gpt-5-mini", temp=0.2)
        self.assertTrue(ok, msg=dbg)
        self.assertFalse(df.empty)
        self.assertIn("total_prospects", df.columns)
        self.assertEqual(int(df["total_prospects"].iloc[0]), 120)


if __name__ == "__main__":
    unittest.main()
