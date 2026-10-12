"""Report rendering for the Clinical Screening Assistant.

Renders structured Report models into Markdown and printable HTML.
"""
from __future__ import annotations

import html

from typing import Union

from .state import Report


def render_report_markdown(report: Union[Report, dict]) -> str:
    """Render a Report model or dict into clinician-ready Markdown."""
    if isinstance(report, dict):
        rep = Report.model_validate(report)
    else:
        rep = report

    intake = rep.intake
    risk_display = rep.final_risk.upper()

    lines: list[str] = [
        "# Clinical Screening Summary Report",
        "",
        f"**Overall Assessment Level:** `{risk_display}`",
        f"**Image-Based Assessment:** `{rep.image_risk.upper()}`",
        f"**Model score on melanoma / basal cell / actinic keratosis classes (uncalibrated, not a risk estimate):** `{rep.concern_probability * 100:.1f}%`",
        "",
    ]

    # Red flags
    lines.append("## Patient-Reported Red Flags")
    if rep.red_flags:
        for rf in rep.red_flags:
            lines.append(f"- **WARNING:** {rf}")
    else:
        lines.append("- *No acute red-flag criteria identified in patient questionnaire.*")
    lines.append("")

    # Patient reported symptoms / intake
    lines.append("## Patient-Reported Clinical History")
    lines.append(f"- **Duration:** {intake.duration_weeks} weeks" if intake.duration_weeks is not None else "- **Duration:** Not reported")
    lines.append(f"- **Evolution / Progression:** {intake.evolution or 'Not reported'}")
    symptoms_str = ", ".join(intake.symptoms) if intake.symptoms else "None reported"
    lines.append(f"- **Reported Symptoms:** {symptoms_str}")
    lines.append(f"- **Anatomical Site:** {intake.anatomy or 'Not reported'}")
    lines.append(f"- **Age Bracket:** {intake.age_band or 'Not reported'}")
    lines.append(f"- **Personal History of Skin Cancer:** {intake.previous_skin_cancer or 'Not reported'}")
    lines.append(f"- **Sun Exposure:** {intake.sun_exposure or 'Not reported'}")
    if intake.visual_descriptors:
        lines.append(f"- **Visual Descriptors Reported:** {', '.join(intake.visual_descriptors)}")
    lines.append("")

    # Vision findings
    lines.append("## Vision Model Findings")
    lines.append("| Class Code | Closest image match (not a diagnosis) | Calibrated Probability |")
    lines.append("| :--- | :--- | :--- |")
    for cls in rep.top_classes:
        name = cls.get("name", "Unknown")
        code = cls.get("code", "")
        prob = cls.get("calibrated_probability", cls.get("probability", 0.0))
        lines.append(f"| `{code}` | {name} | {prob * 100:.1f}% |")
    lines.append("")

    # Optional generated summary
    if rep.generated_summary:
        lines.append("## Automated Clinical Summary")
        lines.append(rep.generated_summary)
        lines.append("")

    # Limitations & Performance
    lines.append("## Tool Limitations & Intended Use")
    lines.append(rep.limitations)
    lines.append("")

    if rep.performance_note:
        lines.append("## Model Performance Note")
        lines.append(rep.performance_note)
        lines.append("")

    # Disclaimer
    lines.append("## Important Medical Disclaimer")
    lines.append(f"> {rep.disclaimer}")
    lines.append("")

    return "\n".join(lines)


def render_report_html(report: Union[Report, dict]) -> str:
    """Render a Report model or dict into print-ready standalone HTML."""
    if isinstance(report, dict):
        rep = Report.model_validate(report)
    else:
        rep = report

    intake = rep.intake
    risk_color = {
        "low": "#2e7d32",
        "elevated": "#ed6c02",
        "high": "#d32f2f",
    }.get(rep.final_risk, "#333333")

    red_flags_html = ""
    if rep.red_flags:
        red_flags_html = "<ul>" + "".join(f"<li class='red-flag'><strong>Warning:</strong> {html.escape(str(rf))}</li>" for rf in rep.red_flags) + "</ul>"
    else:
        red_flags_html = "<p class='muted'>No acute red-flag criteria identified in patient questionnaire.</p>"

    table_rows = ""
    for cls in rep.top_classes:
        name = cls.get("name", "Unknown")
        code = cls.get("code", "")
        prob = cls.get("calibrated_probability", cls.get("probability", 0.0))
        table_rows += f"<tr><td><code>{html.escape(str(code))}</code></td><td>{html.escape(str(name))}</td><td><strong>{prob * 100:.1f}%</strong></td></tr>"

    summary_section = ""
    if rep.generated_summary:
        summary_section = f"""
        <section class="section">
            <h2>Automated Summary</h2>
            <p>{html.escape(rep.generated_summary)}</p>
        </section>
        """

    heatmap_section = ""
    if rep.heatmap_data_url:
        heatmap_section = f"""
        <div class="heatmap-container">
            <h4>Model attention (Grad-CAM). Not a lesion outline and not a diagnosis.</h4>
            <img src="{html.escape(rep.heatmap_data_url, quote=True)}" alt="Grad-CAM Heatmap" class="heatmap-img" />
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Clinical Screening Report</title>
<style>
    @page {{
        margin: 20mm;
        size: A4 portrait;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #111;
        line-height: 1.5;
        margin: 0;
        padding: 24px;
        background: #fff;
    }}
    .header {{
        border-bottom: 2px solid #222;
        padding-bottom: 12px;
        margin-bottom: 20px;
    }}
    .header h1 {{
        margin: 0 0 6px 0;
        font-size: 24px;
        letter-spacing: -0.5px;
    }}
    .risk-banner {{
        display: inline-block;
        padding: 6px 14px;
        border-radius: 4px;
        color: #fff;
        background-color: {risk_color};
        font-weight: 700;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 6px;
    }}
    .section {{
        margin-bottom: 20px;
    }}
    .section h2 {{
        font-size: 16px;
        border-bottom: 1px solid #ddd;
        padding-bottom: 4px;
        margin-bottom: 10px;
        color: #222;
    }}
    .grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
    }}
    .red-flag {{
        color: #c62828;
        font-weight: 600;
    }}
    .muted {{
        color: #666;
        font-style: italic;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 8px;
    }}
    th, td {{
        text-align: left;
        padding: 8px 10px;
        border-bottom: 1px solid #eee;
        font-size: 14px;
    }}
    th {{
        background: #f8f8f8;
        font-weight: 600;
    }}
    .disclaimer-box {{
        background: #fff9e6;
        border-left: 4px solid #f59e0b;
        padding: 12px;
        margin-top: 24px;
        font-size: 12px;
        color: #78350f;
    }}
    .heatmap-img {{
        max-width: 220px;
        border-radius: 4px;
        border: 1px solid #ddd;
    }}
    .actions {{
        margin-bottom: 20px;
    }}
    @media print {{
        .actions {{
            display: none;
        }}
        body {{
            padding: 0;
        }}
    }}
</style>
</head>
<body>
<div class="actions">
    <button onclick="window.print()" style="padding: 8px 16px; font-size: 14px; cursor: pointer; border-radius: 4px; border: 1px solid #888; background: #fafafa;">Print or Save as PDF</button>
</div>
<div class="header">
    <h1>Clinical Screening Summary Report</h1>
    <div>
        Risk Level: <span class="risk-banner">{html.escape(str(rep.final_risk))}</span>
        &nbsp;&bull;&nbsp; Model score on melanoma / basal cell / actinic keratosis (uncalibrated): <strong>{rep.concern_probability * 100:.1f}%</strong>
    </div>
</div>

<div class="section">
    <h2>Patient-Reported Red Flags</h2>
    {red_flags_html}
</div>

<div class="section">
    <h2>Patient History</h2>
    <div class="grid">
        <div><strong>Duration:</strong> {f"{intake.duration_weeks} weeks" if intake.duration_weeks is not None else "Not reported"}</div>
        <div><strong>Evolution:</strong> {html.escape(str(intake.evolution or "Not reported"))}</div>
        <div><strong>Symptoms:</strong> {html.escape(", ".join(map(str, intake.symptoms))) if intake.symptoms else "None reported"}</div>
        <div><strong>Location:</strong> {html.escape(str(intake.anatomy or "Not reported"))}</div>
        <div><strong>Age:</strong> {html.escape(str(intake.age_band or "Not reported"))}</div>
        <div><strong>Skin Cancer History:</strong> {html.escape(str(intake.previous_skin_cancer or "Not reported"))}</div>
        <div><strong>Sun Exposure:</strong> {html.escape(str(intake.sun_exposure or "Not reported"))}</div>
    </div>
</div>

<div class="section">
    <h2>Vision Model Findings</h2>
    <table>
        <thead>
            <tr><th>Code</th><th>Closest image match (not a diagnosis)</th><th>Calibrated Probability</th></tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
    {heatmap_section}
</div>

{summary_section}

<div class="section">
    <h2>Limitations</h2>
    <p style="font-size: 13px; color: #444;">{html.escape(rep.limitations)}</p>
</div>

<div class="disclaimer-box">
    <strong>Medical Disclaimer:</strong> {html.escape(rep.disclaimer)}
</div>
</body>
</html>
"""
