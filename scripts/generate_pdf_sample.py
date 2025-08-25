from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas


def create_sample_pdf(path: str = "sample.pdf") -> None:
	c = canvas.Canvas(path, pagesize=LETTER)
	width, height = LETTER
	textobject = c.beginText(72, height - 72)
	lines = [
		"Employee Timesheets (Demo)",
		"",
		"Alice Johnson | 2025-01-03 | Website Redesign | 8h | Wireframes and stakeholder review",
		"Bob Smith | 2025-01-04 | Data Pipeline | 7h | ETL job fixes and monitoring",
		"Carol Lee | 2025-01-05 | Mobile App | 6h | Push notification setup and testing",
		"David Kim | 2025-01-06 | Analytics | 8h | Dashboard funnel metrics and QA",
		"Emma Davis | 2025-01-07 | API | 7h | Rate limit middleware and docs",
		"Frank Miller | 2025-01-03 | DevOps | 8h | CI workflow caching and images",
		"Grace Wilson | 2025-01-04 | CRM | 6h | Contact import mapping and dedupe",
		"Henry Clark | 2025-01-05 | Security | 8h | 2FA enrollment flow and logs",
		"Irene Patel | 2025-01-06 | SSO | 7h | SAML metadata config and testing",
		"Jack Nguyen | 2025-01-07 | Integrations | 8h | Slack app message actions",
		"Karen Lopez | 2025-01-03 | Docs | 6h | User guide updates and examples",
		"Liam Turner | 2025-01-04 | Observability | 8h | API logs and traces filters",
		"Maya Chen | 2025-01-05 | Billing | 7h | Invoice PDF layout tweaks",
		"Noah Brown | 2025-01-06 | Search | 8h | TF-IDF tuning and tests",
		"Olivia Park | 2025-01-07 | Notifications | 6h | Email templates and retry logic",
		"Peter Zhang | 2025-01-03 | Data Export | 8h | CSV export filters and timezone",
		"Quinn Rivera | 2025-01-04 | Access Control | 7h | Role matrix and permission checks",
		"Ruby Singh | 2025-01-05 | Webhooks | 8h | Retry/backoff and signature docs",
		"Sam Walker | 2025-01-06 | Migration | 7h | Mapping templates and validation",
		"Tina Gomez | 2025-01-07 | Support | 6h | FAQ curation and macro updates",
	]
	for line in lines:
		textobject.textLine(line)
	c.drawText(textobject)
	c.showPage()
	c.save()


if __name__ == "__main__":
	create_sample_pdf()
