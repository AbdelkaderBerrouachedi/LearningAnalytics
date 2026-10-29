"""
Actionable Pedagogical Intervention Engine.
Translates individual student risk probabilities and SHAP attribution vectors
into concrete, theory-grounded intervention tickets mapped to Table 10.
"""
from typing import Dict, List, Any
import numpy as np
import pandas as pd

class PedagogicalInterventionEngine:
    """
    Automated triage system implementing the 3-tiered intervention catalog.
    Bridges predictive learning analytics with evidence-based instructional support.
    """
    def __init__(self, top_budget_pct: float = 0.20):
        self.top_budget_pct = top_budget_pct

    def generate_intervention_plan(
        self,
        student_row: pd.Series,
        predicted_risk: float,
        is_in_budget: bool
    ) -> Dict[str, Any]:
        """
        Evaluate student behavioral metrics and generate tailored pedagogical intervention tickets.
        """
        days_inactive = float(student_row.get("days_since_last_click", 0))
        missed_tma = int(student_row.get("missed_assessments", 0))
        velocity_ratio = float(student_row.get("click_ratio_7d_to_14d", 1.0))
        total_clicks = float(student_row.get("total_clicks", 0))

        assigned_tiers = []

        # Tier 2 Check (Highest Urgency): Assessment Non-Compliance
        if missed_tma >= 1 and is_in_budget:
            assigned_tiers.append({
                "tier": "Tier 2: Intensive Advisor Outreach",
                "urgency": "High (Within 48h)",
                "theoretical_grounding": "Tinto's Student Departure & Academic Momentum Theory",
                "dominant_signal": f"Missed {missed_tma} scheduled summative assessment(s)",
                "prescribed_action": (
                    "Trigger personal advisor telephone/email outreach. Conduct academic diagnosis, "
                    "grant assignment grace periods if extenuating circumstances exist, and scaffold deadline milestones."
                ),
                "field_evidence": "Rust & Motz (2020) RCT; Alalawi et al. (2024)"
            })

        # Tier 1 Check: Behavioral Inactivity & Disrupted Habit Loop
        if days_inactive > 7 and is_in_budget:
            assigned_tiers.append({
                "tier": "Tier 1: Automated Digital Nudge",
                "urgency": "Medium (Automated immediate)",
                "theoretical_grounding": "Zimmerman's Self-Regulated Learning (SRL) & Choice Architecture",
                "dominant_signal": f"Prolonged VLE dormancy: {days_inactive:.0f} days since last interaction",
                "prescribed_action": (
                    "Issue automated, empathetic email/push nudge containing a weekly study planning checklist, "
                    "single-click access to the current week's discussion forum, and micro-task summaries."
                ),
                "field_evidence": "Shi et al. (2024); Gatare et al. (2023)"
            })

        # Tier 3 Check: Interaction Velocity Collapse / Conceptual Impasse
        if velocity_ratio < 0.35 and total_clicks > 50 and is_in_budget:
            assigned_tiers.append({
                "tier": "Tier 3: Adaptive Instructional Scaffolding",
                "urgency": "Medium (Curricular recommendation)",
                "theoretical_grounding": "Cognitive Load Theory (Sweller, 2011) & Instructional Design Loop Closure",
                "dominant_signal": f"Decelerating clickstream velocity: 7d/14d ratio = {velocity_ratio:.2f} (< 0.35)",
                "prescribed_action": (
                    "Recommend modular remedial video walkthroughs, peer-assisted study sessions, and "
                    "formative concept-check quizzes to resolve conceptual stumbling blocks."
                ),
                "field_evidence": "Yamada et al. (2026); Yu (2024)"
            })

        # Default monitoring if risk is below budget threshold
        if not assigned_tiers:
            if is_in_budget:
                assigned_tiers.append({
                    "tier": "General Proactive Monitoring",
                    "urgency": "Low",
                    "theoretical_grounding": "Institutional Early Warning Support",
                    "dominant_signal": f"Cumulative predicted risk {predicted_risk*100:.1f}%",
                    "prescribed_action": "Standard LMS dashboard alert; maintain passive monitoring.",
                    "field_evidence": "Ba\u00f1eres et al. (2023)"
                })
            else:
                assigned_tiers.append({
                    "tier": "Standard Curriculum Track",
                    "urgency": "None",
                    "theoretical_grounding": "Satisfactory Self-Regulated Engagement",
                    "dominant_signal": "Behavior within baseline parameters",
                    "prescribed_action": "No intervention required.",
                    "field_evidence": "N/A"
                })

        return {
            "id_student": int(student_row.get("id_student", 0)),
            "code_module": str(student_row.get("code_module", "")),
            "code_presentation": str(student_row.get("code_presentation", "")),
            "predicted_risk": float(predicted_risk),
            "flagged_for_intervention": bool(is_in_budget),
            "interventions": assigned_tiers
        }

    def triage_cohort(
        self,
        df_cohort: pd.DataFrame,
        predicted_probs: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Rank cohort by calibrated withdrawal risk and generate triage plans
        under finite advisor contact capacity.
        """
        n_students = len(df_cohort)
        k_cutoff = max(1, int(np.floor(self.top_budget_pct * n_students)))
        ranked_indices = np.argsort(predicted_probs)[::-1]
        top_k_set = set(ranked_indices[:k_cutoff])

        triage_results = []
        for idx in range(n_students):
            is_budget = idx in top_k_set
            plan = self.generate_intervention_plan(
                df_cohort.iloc[idx],
                predicted_probs[idx],
                is_budget
            )
            triage_results.append(plan)

        return triage_results
