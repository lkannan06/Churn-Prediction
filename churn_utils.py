import numpy as np
import pandas as pd

ENGAGEMENT_COLS = ["Usage_Frequency", "Feature_Usage_Count",
                   "Avg_Session_Duration_Min", "Email_Open_Rate_Pct"]


def engineer_features(raw: pd.DataFrame, meta: dict) -> pd.DataFrame:
    d = raw.copy()

    if "Signup_Date" in d.columns:
        ref = pd.to_datetime(meta["ref_date"])
        d["Days_Since_Signup"] = (ref - pd.to_datetime(d["Signup_Date"])).dt.days.clip(lower=0)
    elif "Days_Since_Signup" not in d.columns:
        raise ValueError("The Signup_Date column (or Days_Since_Signup) is required.")

    tenure = d["Tenure_Months"].replace(0, 1)

    d["Avg_Monthly_Spend"] = d["Total_Charges"] / tenure
    d["Billing_Discrepancy"] = d["Avg_Monthly_Spend"] - d["Monthly_Charges"]
    d["CLV_Proxy"] = d["Monthly_Charges"] * d["Tenure_Months"]

    mx = meta["engagement_max"]
    d["Engagement_Score"] = sum(d[c] / mx[c] for c in ENGAGEMENT_COLS) / len(ENGAGEMENT_COLS)
    d["Is_Inactive"] = (d["Days_Since_Last_Activity"] > meta["inactive_median"]).astype(int)

    d["Complaint_Rate"] = d["Complaint_Count"] / tenure
    d["Late_Payment_Rate"] = d["Late_Payment_Count"] / tenure
    d["Support_Burden"] = d["Support_Tickets_Raised"] + d["Complaint_Count"]
    d["Resolution_Efficiency"] = d["Avg_Resolution_Time_Hrs"] / (d["Support_Tickets_Raised"] + 1)
    d["Satisfaction_Support_Ratio"] = d["Satisfaction_Score"] / (d["Support_Tickets_Raised"] + 1)

    d["Loyalty_Tenure"] = d["Loyalty_Program_Member"] * d["Tenure_Months"]
    d["Has_Referral"] = (d["Referral_Count"] > 0).astype(int)
    d["Discount_Flag"] = (d["Discount_Applied_Pct"] > 0).astype(int)

    d["Tenure_Group"] = pd.cut(d["Tenure_Months"], bins=[-1, 12, 36, 100],
                               labels=["New (0-12mo)", "Mid (13-36mo)", "Long-term (36mo+)"]).astype(str)
    d["Satisfaction_Group"] = pd.cut(d["Satisfaction_Score"], bins=[-1, 4, 7, 10],
                                     labels=["Low", "Medium", "High"]).astype(str)

    return d[meta["model_columns"]]


def predict_frame(bundle: dict, meta: dict, raw: pd.DataFrame):
    feats = engineer_features(raw, meta)
    proba = bundle["model"].predict_proba(feats)[:, 1]
    pred = (proba >= bundle["threshold"]).astype(int)
    return proba, pred


def risk_level(proba: float, threshold: float) -> str:
    if proba >= threshold:
        return "High"
    if proba >= threshold * 0.5:
        return "Medium"
    return "Low"
