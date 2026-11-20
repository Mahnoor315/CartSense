import numpy as np
import pandas as pd

MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]


EXAMPLES = {
    "likely_buyer": dict(pages=45, minutes=22.0, bounce=0.2, exit_=1.5, page_value=35.0,
                         visitor="Returning_Visitor", month="Nov", weekend=False, special=0.0),
    "window_shopper": dict(pages=3, minutes=0.7, bounce=8.0, exit_=14.0, page_value=0.0,
                           visitor="New_Visitor", month="Mar", weekend=True, special=0.0),
}


def default_row(X_reference: pd.DataFrame) -> dict:
    """Median for numeric columns, most common value for the rest (used for fields the form does not ask for)."""
    row = {}
    for col in X_reference.columns:
        s = X_reference[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            row[col] = s.median()
        else:
            row[col] = s.mode().iloc[0]
    return row


def build_row(X_reference, pages, minutes, bounce, exit_, page_value, visitor, month, weekend, special):
    """Turn form inputs into a one-row DataFrame with exactly the training columns."""
    row = default_row(X_reference)
    row.update({
        "ProductRelated": pages,
        "ProductRelated_Duration": minutes * 60.0,   # dataset stores seconds
        "BounceRates": bounce / 100.0,               # dataset stores fractions
        "ExitRates": exit_ / 100.0,
        "PageValues": page_value,
        "VisitorType": visitor,
        "Month": month,
        "Weekend": bool(weekend),
        "SpecialDay": special,
    })
    return pd.DataFrame([row], columns=X_reference.columns)


def band(prob: float):
    """Probability band -> (label, colour key, suggested marketing action). Bands are illustrative."""
    if prob >= 0.60:
        return "High intent", "high", "Already likely to buy. Do not discount. Keep checkout fast and show delivery info."
    if prob >= 0.30:
        return "Medium intent", "mid", "On the fence. Good moment for a chat prompt or a small, time-limited offer."
    return "Low intent", "low", "Browsing only. Suggest a popular or related product before they leave."


def local_explanation(rf_pipeline, row: pd.DataFrame, top_n: int = 6):
    """Per-session SHAP values from the Random Forest (same method as your SHAP cell).
    Returns a DataFrame [Feature, Impact] sorted by |Impact|, or None if shap is not installed."""
    try:
        import shap
    except ImportError:
        return None
    pre = rf_pipeline.named_steps["preprocessor"]
    Xt = pre.transform(row)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt, dtype=np.float64)
    names = pre.get_feature_names_out()
    sv = shap.TreeExplainer(rf_pipeline.named_steps["classifier"]).shap_values(Xt)
    if isinstance(sv, list):
        vals = sv[1][0]
    elif np.ndim(sv) == 3:
        vals = sv[0, :, 1]
    else:
        vals = sv[0]
    clean = [n.replace("num__", "").replace("cat__", "") for n in names]
    df = pd.DataFrame({"Feature": clean, "Impact": vals})
    return df.reindex(df["Impact"].abs().sort_values(ascending=False).index).head(top_n)
