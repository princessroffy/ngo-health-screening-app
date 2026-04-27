from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import io

from datetime import datetime
from pathlib import Path

import joblib
from flask import (
    Flask,
    Response,
    abort,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_
from werkzeug.security import check_password_hash, generate_password_hash

from config import Config

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "health_risk_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

FEATURE_COLUMNS = [
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "blood_sugar",
    "cholesterol",
    "activity_level",
    "smoking",
    "family_history",
]

app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access the NGO screening system."


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="staff", nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Beneficiary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20))
    phone = db.Column(db.String(30))
    state = db.Column(db.String(100))
    community = db.Column(db.String(150))
    outreach_event = db.Column(db.String(150))
    project_name = db.Column(db.String(150))
    funding_source = db.Column(db.String(150))
    screening_date = db.Column(db.String(50))
    consent = db.Column(db.String(10), default="no")
    occupation = db.Column(db.String(120))
    marital_status = db.Column(db.String(50))
    household_size = db.Column(db.Integer)
    vulnerable_group = db.Column(db.String(120))
    created_by = db.Column(db.Integer, db.ForeignKey("user.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    creator = db.relationship("User", foreign_keys=[created_by])
    screenings = db.relationship(
        "Screening",
        backref="beneficiary",
        cascade="all, delete-orphan",
        lazy=True,
    )


class Screening(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    beneficiary_id = db.Column(
        db.Integer,
        db.ForeignKey("beneficiary.id"),
        nullable=False,
    )
    bmi = db.Column(db.Float, nullable=False)
    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    blood_sugar = db.Column(db.Float, nullable=False)
    cholesterol = db.Column(db.Float, nullable=False)
    activity_level = db.Column(db.Float, nullable=False)
    smoking = db.Column(db.Integer, nullable=False)
    family_history = db.Column(db.Integer, nullable=False)
    risk_score = db.Column(db.Float)
    risk_level = db.Column(db.String(20))
    summary = db.Column(db.Text)
    referral_status = db.Column(db.String(80), default="Not Referred")
    follow_up_status = db.Column(db.String(50), default="Pending")
    notes = db.Column(db.Text)
    screened_by = db.Column(db.Integer, db.ForeignKey("user.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    screener = db.relationship("User", foreign_keys=[screened_by])


class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    action = db.Column(db.String(120), nullable=False)
    details = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User")


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def create_audit_log(action, details=""):
    user_id = current_user.id if current_user.is_authenticated else None
    db.session.add(AuditLog(user_id=user_id, action=action, details=details))
    db.session.commit()


def safe_int(value):
    if value in (None, ""):
        return None
    return int(value)


def parse_float(field_name):
    value = request.form.get(field_name, "").strip()
    if not value:
        raise ValueError(
            f"{field_name.replace('_', ' ').title()} is required.")
    return float(value)


def activity_name(value):
    labels = {
        0: "Low",
        1: "Moderate",
        2: "High",
    }
    return labels.get(int(value), "Unknown")


def risk_badge_class(level):
    return {
        "Low": "risk-low",
        "Moderate": "risk-moderate",
        "High": "risk-high",
    }.get(level, "risk-unknown")


def referral_for_risk(level):
    if level == "Low":
        return "Health Education Only"
    if level == "Moderate":
        return "Monitor and Follow Up"
    return "Refer to Health Facility"


def summary_for_risk(level):
    if level == "Low":
        return (
            "Screening indicators suggest lower immediate risk. Provide health "
            "education and encourage routine checks."
        )
    if level == "Moderate":
        return (
            "Some indicators need attention. Recommend monitoring, counselling, "
            "and follow-up contact."
        )
    return (
        "Several indicators suggest elevated risk. Recommend prompt referral "
        "to a qualified health facility."
    )


def rule_based_prediction(values):
    points = 0
    points += 1.2 if values["bmi"] >= 30 else 0
    points += 1.7 if values["systolic_bp"] >= 140 else 0
    points += 1.2 if values["diastolic_bp"] >= 90 else 0
    points += 1.7 if values["blood_sugar"] >= 126 else 0
    points += 1.3 if values["cholesterol"] >= 240 else 0
    points += 0.8 if values["activity_level"] == 0 else 0
    points += 1.0 if values["smoking"] == 1 else 0
    points += 0.9 if values["family_history"] == 1 else 0

    if points >= 4.5:
        level = "High"
    elif points >= 2.2:
        level = "Moderate"
    else:
        level = "Low"

    score = min(100, max(1, round(20 + points * 15, 1)))
    return score, level


def predict_risk(values):
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return rule_based_prediction(values)

    import pandas as pd

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    input_frame = pd.DataFrame(
        [[values[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    scaled_input = scaler.transform(input_frame)
    level = model.predict(scaled_input)[0]

    if hasattr(model, "predict_proba"):
        weights = {"Low": 25, "Moderate": 60, "High": 90}
        probabilities = model.predict_proba(scaled_input)[0]
        score = sum(
            probabilities[index] * weights.get(label, 50)
            for index, label in enumerate(model.classes_)
        )
        return round(score, 1), level

    return rule_based_prediction(values)


def screening_scope_query():
    query = Screening.query.join(Beneficiary)
    if current_user.role != "admin":
        query = query.filter(
            or_(
                Screening.screened_by == current_user.id,
                Beneficiary.created_by == current_user.id,
            )
        )
    return query


def can_view_screening(screening):
    return (
        current_user.role == "admin"
        or screening.screened_by == current_user.id
        or screening.beneficiary.created_by == current_user.id
    )


def can_view_beneficiary(beneficiary):
    return current_user.role == "admin" or beneficiary.created_by == current_user.id


def aggregate_counts(records, key_getter):
    counts = {}
    for record in records:
        key = key_getter(record) or "Unspecified"
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)


def apply_record_filters(query):
    date = request.args.get("date", "").strip()
    location = request.args.get("location", "").strip()
    risk_level = request.args.get("risk_level", "").strip()
    outreach_event = request.args.get("outreach_event", "").strip()
    follow_up_status = request.args.get("follow_up_status", "").strip()

    if date:
        query = query.filter(Beneficiary.screening_date == date)
    if location:
        query = query.filter(Beneficiary.community.ilike(f"%{location}%"))
    if risk_level:
        query = query.filter(Screening.risk_level == risk_level)
    if outreach_event:
        query = query.filter(
            Beneficiary.outreach_event.ilike(f"%{outreach_event}%"))
    if follow_up_status:
        query = query.filter(Screening.follow_up_status == follow_up_status)

    return query


@app.context_processor
def inject_helpers():
    return {
        "activity_name": activity_name,
        "risk_badge_class": risk_badge_class,
        "now": datetime.utcnow(),
    }


@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        login_user(user)
        create_audit_log("login", f"{user.email} logged in")
        flash(f"Welcome back, {user.name}.", "success")

        next_page = request.args.get("next")
        if next_page and next_page.startswith("/"):
            return redirect(next_page)
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    create_audit_log("logout", f"{current_user.email} logged out")
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    records = screening_scope_query().order_by(Screening.created_at.desc()).all()
    total = len(records)
    low_count = sum(1 for record in records if record.risk_level == "Low")
    moderate_count = sum(
        1 for record in records if record.risk_level == "Moderate")
    high_count = sum(1 for record in records if record.risk_level == "High")
    referrals_made = sum(
        1 for record in records if record.referral_status == "Refer to Health Facility"
    )
    pending_followups = sum(
        1 for record in records if record.follow_up_status == "Pending"
    )

    by_location = aggregate_counts(
        records, lambda item: item.beneficiary.community)[:6]
    by_event = aggregate_counts(
        records, lambda item: item.beneficiary.outreach_event)[:6]
    recent_screenings = records[:8]

    return render_template(
        "dashboard.html",
        total=total,
        low_count=low_count,
        moderate_count=moderate_count,
        high_count=high_count,
        referrals_made=referrals_made,
        pending_followups=pending_followups,
        by_location=by_location,
        by_event=by_event,
        recent_screenings=recent_screenings,
    )


@app.route("/beneficiary/new", methods=["GET", "POST"])
@login_required
def register_beneficiary():
    if request.method == "POST":
        consent = request.form.get("consent", "no")
        if consent != "yes":
            flash("Consent is required before beneficiary data can be saved.", "error")
            return render_template("register_beneficiary.html")

        try:
            beneficiary = Beneficiary(
                full_name=request.form.get("full_name", "").strip(),
                age=safe_int(request.form.get("age")),
                gender=request.form.get("gender", "").strip(),
                phone=request.form.get("phone", "").strip(),
                state=request.form.get("state", "").strip(),
                community=request.form.get("community", "").strip(),
                outreach_event=request.form.get("outreach_event", "").strip(),
                project_name=request.form.get("project_name", "").strip(),
                funding_source=request.form.get("funding_source", "").strip(),
                screening_date=request.form.get("screening_date")
                or datetime.utcnow().date().isoformat(),
                consent=consent,
                occupation=request.form.get("occupation", "").strip(),
                marital_status=request.form.get("marital_status", "").strip(),
                household_size=safe_int(request.form.get("household_size")),
                vulnerable_group=request.form.get(
                    "vulnerable_group", "").strip(),
                created_by=current_user.id,
            )
        except ValueError:
            flash("Please enter valid numbers for age and household size.", "error")
            return render_template("register_beneficiary.html")

        if not beneficiary.full_name or beneficiary.age is None:
            flash("Full name and age are required.", "error")
            return render_template("register_beneficiary.html")

        db.session.add(beneficiary)
        db.session.commit()
        create_audit_log(
            "beneficiary_created",
            f"Beneficiary #{beneficiary.id} registered by {current_user.email}",
        )
        flash("Beneficiary registered. Continue to screening.", "success")
        return redirect(url_for("new_screening", beneficiary_id=beneficiary.id))

    return render_template("register_beneficiary.html")


@app.route("/screening/new/<int:beneficiary_id>", methods=["GET", "POST"])
@login_required
def new_screening(beneficiary_id):
    beneficiary = db.session.get(Beneficiary, beneficiary_id)
    if not beneficiary:
        abort(404)
    if not can_view_beneficiary(beneficiary):
        abort(403)

    if request.method == "POST":
        try:
            values = {
                "bmi": parse_float("bmi"),
                "systolic_bp": parse_float("systolic_bp"),
                "diastolic_bp": parse_float("diastolic_bp"),
                "blood_sugar": parse_float("blood_sugar"),
                "cholesterol": parse_float("cholesterol"),
                "activity_level": float(request.form.get("activity_level", "1")),
                "smoking": int(request.form.get("smoking", "0")),
                "family_history": int(request.form.get("family_history", "0")),
            }
        except ValueError as exc:
            flash(str(exc), "error")
            return render_template("screening_form.html", beneficiary=beneficiary)

        risk_score, risk_level = predict_risk(values)
        screening = Screening(
            beneficiary_id=beneficiary.id,
            bmi=values["bmi"],
            systolic_bp=values["systolic_bp"],
            diastolic_bp=values["diastolic_bp"],
            blood_sugar=values["blood_sugar"],
            cholesterol=values["cholesterol"],
            activity_level=values["activity_level"],
            smoking=values["smoking"],
            family_history=values["family_history"],
            risk_score=risk_score,
            risk_level=risk_level,
            summary=summary_for_risk(risk_level),
            referral_status=referral_for_risk(risk_level),
            follow_up_status=request.form.get("follow_up_status", "Pending"),
            notes=request.form.get("notes", "").strip(),
            screened_by=current_user.id,
        )

        db.session.add(screening)
        db.session.commit()
        create_audit_log(
            "screening_created",
            f"Screening #{screening.id} saved for beneficiary #{beneficiary.id}",
        )
        return redirect(url_for("screening_result", screening_id=screening.id))

    return render_template("screening_form.html", beneficiary=beneficiary)


@app.route("/screening/result/<int:screening_id>")
@login_required
def screening_result(screening_id):
    screening = db.session.get(Screening, screening_id)
    if not screening:
        abort(404)
    if not can_view_screening(screening):
        abort(403)

    return render_template("screening_result.html", screening=screening)


@app.route("/records")
@login_required
def records():
    query = apply_record_filters(screening_scope_query())
    screenings = query.order_by(Screening.created_at.desc()).all()
    active_filters = {
        key: value
        for key, value in request.args.to_dict().items()
        if value
    }
    export_url = url_for("export_csv", **active_filters)

    return render_template(
        "records.html",
        screenings=screenings,
        filters=request.args,
        export_url=export_url,
    )


@app.route("/screening/<int:screening_id>/follow-up", methods=["POST"])
@login_required
def update_follow_up(screening_id):
    screening = db.session.get(Screening, screening_id)
    if not screening:
        abort(404)
    if not can_view_screening(screening):
        abort(403)

    screening.referral_status = request.form.get(
        "referral_status",
        screening.referral_status,
    )
    screening.follow_up_status = request.form.get(
        "follow_up_status",
        screening.follow_up_status,
    )
    notes = request.form.get("notes", "").strip()
    if notes:
        screening.notes = notes

    db.session.commit()
    create_audit_log(
        "follow_up_updated",
        f"Screening #{screening.id} follow-up updated",
    )
    flash("Follow-up details updated.", "success")
    return redirect(request.referrer or url_for("records"))


@app.route("/export/csv")
@login_required
def export_csv():
    if current_user.role != "admin":
        abort(403)

    import pandas as pd

    query = apply_record_filters(screening_scope_query())
    screenings = query.order_by(Screening.created_at.desc()).all()

    rows = []
    for item in screenings:
        beneficiary = item.beneficiary
        rows.append(
            {
                "beneficiary_name": beneficiary.full_name,
                "age": beneficiary.age,
                "gender": beneficiary.gender,
                "phone": beneficiary.phone,
                "state": beneficiary.state,
                "community": beneficiary.community,
                "outreach_event": beneficiary.outreach_event,
                "project_name": beneficiary.project_name,
                "funding_source": beneficiary.funding_source,
                "screening_date": beneficiary.screening_date,
                "risk_level": item.risk_level,
                "risk_score": item.risk_score,
                "referral_status": item.referral_status,
                "follow_up_status": item.follow_up_status,
                "screening_officer": item.screener.name if item.screener else "",
                "created_at": item.created_at.isoformat(timespec="seconds"),
            }
        )

    columns = [
        "beneficiary_name",
        "age",
        "gender",
        "phone",
        "state",
        "community",
        "outreach_event",
        "project_name",
        "funding_source",
        "screening_date",
        "risk_level",
        "risk_score",
        "referral_status",
        "follow_up_status",
        "screening_officer",
        "created_at",
    ]
    data_frame = pd.DataFrame(rows, columns=columns)
    csv_data = data_frame.to_csv(index=False)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    create_audit_log(
        "csv_exported",
        f"{current_user.email} exported {len(rows)} records",
    )

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=screening_records_{timestamp}.csv"
        },
    )


def seed_user(name, email, password, role):
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return

    db.session.add(
        User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            role=role,
        )
    )
    db.session.commit()


def initialize_database():
    with app.app_context():
        db.create_all()
        seed_user("NGO Admin", "admin@ngo.org", "admin123", "admin")
        seed_user("Field Staff", "staff@ngo.org", "staff123", "staff")


initialize_database()


if __name__ == "__main__":
    app.run(debug=True, port=5001)


@app.route("/screening/pdf/<int:screening_id>")
@login_required
def download_pdf(screening_id):
    screening = Screening.query.get_or_404(screening_id)
    beneficiary = screening.beneficiary

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    content = []

    # Title
    content.append(
        Paragraph("Community Health Screening Report", styles["Title"]))
    content.append(Spacer(1, 0.3 * inch))

    # Beneficiary info
    content.append(
        Paragraph(f"<b>Name:</b> {beneficiary.full_name}", styles["Normal"]))
    content.append(
        Paragraph(f"<b>Age:</b> {beneficiary.age}", styles["Normal"]))
    content.append(
        Paragraph(f"<b>Gender:</b> {beneficiary.gender}", styles["Normal"]))
    content.append(
        Paragraph(f"<b>Phone:</b> {beneficiary.phone}", styles["Normal"]))
    content.append(
        Paragraph(f"<b>Community:</b> {beneficiary.community}", styles["Normal"]))
    content.append(Paragraph(
        f"<b>Outreach Event:</b> {beneficiary.outreach_event}", styles["Normal"]))
    content.append(Spacer(1, 0.3 * inch))

    # Health values
    content.append(
        Paragraph("<b>Health Screening Values</b>", styles["Heading2"]))
    content.append(Paragraph(f"BMI: {screening.bmi}", styles["Normal"]))
    content.append(
        Paragraph(f"Systolic BP: {screening.systolic_bp}", styles["Normal"]))
    content.append(
        Paragraph(f"Diastolic BP: {screening.diastolic_bp}", styles["Normal"]))
    content.append(
        Paragraph(f"Blood Sugar: {screening.blood_sugar}", styles["Normal"]))
    content.append(
        Paragraph(f"Cholesterol: {screening.cholesterol}", styles["Normal"]))
    content.append(Spacer(1, 0.3 * inch))

    # Result
    content.append(Paragraph("<b>Risk Assessment</b>", styles["Heading2"]))
    content.append(
        Paragraph(f"Risk Level: {screening.risk_level}", styles["Normal"]))
    content.append(
        Paragraph(f"Risk Score: {screening.risk_score}%", styles["Normal"]))
    content.append(
        Paragraph(f"Summary: {screening.summary}", styles["Normal"]))
    content.append(Spacer(1, 0.3 * inch))

    # Recommendation
    content.append(Paragraph("<b>Recommendation</b>", styles["Heading2"]))
    content.append(Paragraph(screening.referral_status, styles["Normal"]))
    content.append(Spacer(1, 0.5 * inch))

    # Footer
    content.append(Paragraph(
        "This report is for screening purposes only and does not replace professional medical diagnosis.",
        styles["Italic"]
    ))

    doc.build(content)

    buffer.seek(0)

    return Response(
        buffer,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=screening_{screening.id}.pdf"
        }
    )
