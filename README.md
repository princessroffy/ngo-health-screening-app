# AI-Powered Community Health Screening App for NGOs

This is an NGO-ready Flask MVP for registering beneficiaries, running community health screenings, saving records, tracking referrals/follow-ups, and exporting CSV reports.

## Paste/Create Files in This Exact Order

1. `requirements.txt`
2. `config.py`
3. `train_model.py`
4. `app.py`
5. `templates/base.html`
6. `templates/login.html`
7. `templates/dashboard.html`
8. `templates/register_beneficiary.html`
9. `templates/screening_form.html`
10. `templates/screening_result.html`
11. `templates/records.html`
12. `static/style.css`
13. `static/app.js`
14. `.gitignore`

## Setup

```bash
cd ngo_health_screening_app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```

On Windows:

```bash
cd ngo_health_screening_app
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

## Demo Login

Admin:

```text
Email: admin@ngo.org
Password: admin123
```

Staff:

```text
Email: staff@ngo.org
Password: staff123
```

## Main Routes

- `/login` staff/admin login
- `/dashboard` dashboard metrics
- `/beneficiary/new` beneficiary registration
- `/screening/new/<beneficiary_id>` screening form
- `/screening/result/<screening_id>` screening result
- `/records` records, filters, and follow-up tracking
- `/export/csv` admin CSV export

## Note

Use a secure `SECRET_KEY`, stronger account management, HTTPS, backups, and real validated clinical data before any real deployment.
