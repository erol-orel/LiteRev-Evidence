"""English rendering of the built-in (GESICA) scenario catalogue.

`GESICA_SCENARIO_METADATA` in main.py is written in French and stored in the
database as the system scenarios. The interface has a language toggle, but the
catalogue's titles, descriptions and recommended actions came back in French under
English. This module carries the English versions, keyed by scenario id, and
`localize_gesica()` applies them to a scenario payload when the requested language
is English. The French text stays the source of truth; only the rendering changes.
"""
from __future__ import annotations

from typing import Any

GESICA_EN: dict[str, dict[str, Any]] = {
    "cardiac-arrest-prediction": {
        "title": "Out-of-Hospital Cardiac Arrest (OHCA) Prediction",
        "description": "Spatio-temporal models predicting the incidence of out-of-hospital cardiac arrests (OHCA) from machine learning, circadian rhythms, climate and weather data, to optimise the chain of survival and the preventive positioning of resources.",
        "recommended_actions": [
            "Deploy AI-assisted acoustic detection of agonal breathing (gasping) at the 15/144 dispatch centre",
            "Optimise the coverage and dispatch of citizen first responders equipped with AEDs through dynamic geolocation",
            "Preventively adjust the positioning of mobile intensive care units and resuscitation ambulances in high-OHCA-risk areas",
            "Integrate connected defibrillator (IoT) data for a real-time map of AED accessibility",
        ],
    },
    "stroke-detection": {
        "title": "Prehospital Stroke Detection",
        "description": "Clinical decision-support systems for the early identification of strokes in the field, severity assessment through automated scores (FAST, NIHSS, LVO) and optimal, direct routing to reperfusion centres (thrombolysis/thrombectomy).",
        "recommended_actions": [
            "Integrate automated, AI-guided stroke scales into the paramedics' on-board patient record",
            "Route patients directly, without transit through general emergency departments, to the reference stroke unit (HUG/CHUV)",
            "Trigger an automatic pre-alert for the interventional neuroradiology team when a large-vessel occlusion (LVO) is strongly suspected",
            "Optimise the door-to-needle time through secure prehospital transmission of clinical data",
        ],
    },
    "trauma-severity-assessment": {
        "title": "Trauma Severity Assessment",
        "description": "Predictive models and risk-stratification scores (ISS, RTS, TRISS) for the immediate assessment of severely injured patients (road accidents, falls, mountain injuries), to route them without delay to a trauma centre of the appropriate level.",
        "recommended_actions": [
            "Deploy predictive models of massive transfusion need (shock index, haemorrhage score) from the first field assessment",
            "Systematically route severe trauma (ISS > 15) to an accredited level 1 trauma centre (HUG or CHUV)",
            "Stream vital signs and the FAST ultrasound in real time to the hospital resuscitation bay",
            "Implement damage-control resuscitation protocols guided by decision-support algorithms",
        ],
    },
    "clinical-deterioration-prediction": {
        "title": "Clinical Deterioration Prediction in Transit",
        "description": "Intelligent monitoring of critical patients during ambulance or helicopter transport.",
        "recommended_actions": [
            "Activate deterioration alerts based on multi-parameter vital-sign trends",
            "Prepare advanced resuscitation protocols with the mobile ICU's regulating physician",
            "Adjust the transfer speed or arrange a mobile ICU / helicopter rendezvous when needed",
        ],
    },
    "patient-pathway-optimization": {
        "title": "Cross-Border Patient Pathway Optimisation",
        "description": "Planning patient transfers to the appropriate care facilities while optimising capacity on both sides of the border.",
        "recommended_actions": [
            "Check the real-time availability of specialised beds in France (ROR) and Switzerland",
            "Streamline customs formalities for transfer ambulances",
            "Establish a protocol for return home or nearby follow-up care",
        ],
    },
    "mci-victim-estimation": {
        "title": "Mass-Casualty Incident (MCI) Victim Estimation",
        "description": "Rapid assessment of the number and severity of victims during major events to size the response.",
        "recommended_actions": [
            "Activate the Plan Blanc (FR) / Plan ORCA (CH) in a coordinated way",
            "Use connected triage tools (smart glasses, IoT wristbands) for a real-time inventory",
            "Distribute victim flows evenly across the region's hospitals",
        ],
    },
    "environmental-risk-forecasting": {
        "title": "Environmental Risk Forecasting",
        "description": "Anticipating peaks of air pollution, ozone or allergens and their direct impact on respiratory emergencies.",
        "recommended_actions": [
            "Cross AirGenève and Atmo Auvergne-Rhône-Alpes data with asthma/COPD calls",
            "Send targeted prevention messages to registered vulnerable patients",
            "Anticipate a 15% rise in respiratory-distress calls within 48 hours",
        ],
    },
    "disaster-risk-assessment": {
        "title": "Natural Disaster Risk Assessment",
        "description": "Modelling the health impact of floods, local earthquakes or landslides on EMS infrastructure.",
        "recommended_actions": [
            "Identify stations and ambulance access routes located in flood zones (Arve/Rhône floods)",
            "Establish rescue assembly points outside risk areas",
            "Simulate power or telecommunication outage scenarios",
        ],
    },
    "heatwave-ems-impact": {
        "title": "Heatwave Impact on EMS",
        "description": "Modelling the impact of extreme heat waves on EMS demand and heat-related conditions (heat stroke, hyperthermia).",
        "recommended_actions": [
            "Anticipate a 20-40% rise in EMS calls during heatwave episodes (UTCI > 38°C)",
            "Activate prehospital heat-stroke management protocols",
            "Reinforce crews with rapid cooling equipment (ice packs, misting fans)",
        ],
    },
    "climate-impact-on-ems": {
        "title": "Climate Change Impact on EMS",
        "description": "Long-term and seasonal analysis of how emergency conditions evolve with global warming.",
        "recommended_actions": [
            "Adapt summer duty rosters to more frequent heat waves",
            "Integrate Copernicus climate projections into the cross-border health master plan",
            "Train staff on emerging conditions (vector-borne diseases such as dengue in Europe)",
        ],
    },
    "emergency-call-qualification": {
        "title": "Automated Emergency Call Qualification",
        "description": "Natural-language processing (NLP) and real-time speech-recognition tools assisting medical dispatchers with transcription, automatic detection of clinical keywords and rapid qualification of emergency call reasons.",
        "recommended_actions": [
            "Activate continuous low-latency speech-to-text integrated into the dispatch telephony system",
            "Use specialised NLP models (medical CamemBERT type) to automatically extract clinical entities and key symptoms",
            "Analyse the acoustic features and background noise of the call to detect respiratory distress or panic",
            "Adaptively and dynamically suggest dispatch protocol questions from the first transcribed words",
        ],
    },
    "call-prioritization": {
        "title": "Dispatch Call Prioritisation",
        "description": "Machine-learning algorithms for the dynamic triage and prioritisation of the incoming call queue at the medical dispatch centre, guaranteeing immediate handling of life-threatening emergencies and minimising the risk of undertriage.",
        "recommended_actions": [
            "Automatically move calls identified as suspected cardiac arrest or choking to the absolute top of the queue",
            "Dynamically adjust triage thresholds and queues when the dispatch centre is saturated (call peaks)",
            "Give dispatchers a predictive dashboard of the estimated clinical risk level of each waiting call",
            "Continuously measure prioritisation adequacy to keep undertriage strictly below 5%",
        ],
    },
    "mass-casualty-triage": {
        "title": "Mass-Casualty Triage",
        "description": "Field mass-triage support algorithms to rapidly classify victims (absolute emergency, relative emergency).",
        "recommended_actions": [
            "Apply standardised triage criteria (START/SALT) through a simplified mobile interface",
            "Generate unique QR codes for each victim to track their pathway",
            "Visualise the distribution of severity categories on the advanced medical post map",
        ],
    },
    "undertriage-detection": {
        "title": "Undertriage Detection",
        "description": "Quality-control algorithms to identify severe patients wrongly classified as low priority.",
        "recommended_actions": [
            "Retrospectively analyse dispatch records to identify triage discrepancies",
            "Alert in real time when the recorded vital signs contradict the assigned priority level",
            "Adjust clinical decision trees to bring the undertriage rate below the 5% threshold",
        ],
    },
    "dispatch-decision-support": {
        "title": "Dispatch Decision Support",
        "description": "Expert systems and predictive decision-support models recommending instantly the optimal prehospital resource (emergency ambulance, mobile ICU team, helicopter or on-call general practitioner) according to the suspected clinical severity and the available resources.",
        "recommended_actions": [
            "Automatically suggest sending a cross-border (FR/CH) mobile ICU when its estimated arrival time beats the national resource",
            "Integrate live GPS geolocation and the operational status of vehicles to propose the fastest resource",
            "Suggest alternatives such as out-of-hours GP care, medical advice or non-urgent medical transport for low-severity reasons",
            "Implement a dispatch-adequacy model to reduce unnecessary physician-staffed dispatches (over-dispatch) while keeping patients safe",
        ],
    },
    "triage-support": {
        "title": "Emergency Department Triage Support",
        "description": "Clinical classification algorithms assisting triage nurses in rapidly determining the severity level of emergency department patients on validated scales (Swiss Triage Scale, Rouen Scale), optimising time to care.",
        "recommended_actions": [
            "Automatically compute the theoretical clinical severity level from vital signs and the recorded presenting complaint",
            "Predict at triage the risk of downstream hospitalisation or ICU admission to anticipate patient routing",
            "Generate immediate visual and audible alerts for the triage nurse in case of a major physiological anomaly",
            "Measure inter-observer agreement (Cohen's kappa) between AI-assisted triage and the physician's final assessment",
        ],
    },
    "response-time-optimization": {
        "title": "EMS Response Time Optimisation",
        "description": "Predictive routing models integrating real-time traffic, weather and urban topology to guide emergency vehicles along the fastest route and minimise time to critical care.",
        "recommended_actions": [
            "Compute dynamic emergency routes from real-time traffic congestion data and traffic history",
            "Interface the ambulance navigation system with traffic-light control (green wave) on critical corridors",
            "Specifically model cross-border transit times (Greater Geneva customs, lake bridges) to adapt routes",
            "Continuously evaluate the time-dependent effect of the actual response time on survival in life-threatening emergencies",
        ],
    },
    "ambulance-dispatch-optimization": {
        "title": "Ambulance Fleet Optimisation",
        "description": "Maximal spatio-temporal coverage models (MCLP, DSM) for the preventive and dynamic management and repositioning of the ambulance fleet, guaranteeing optimal territorial coverage according to predicted risks.",
        "recommended_actions": [
            "Dynamically and preventively reposition idle ambulances to fill coverage gaps",
            "Predict hourly high-risk micro-areas for emergency calls to pre-position crews there",
            "Coordinate public, private and volunteer ambulance dispatch transparently on a single platform",
            "Track in real time the share of the target population within 10 minutes of an available ambulance",
        ],
    },
    "staffing-level-prediction": {
        "title": "Required Staffing Forecast",
        "description": "Predictive models to size dispatch teams and ambulance crews according to the expected workload.",
        "recommended_actions": [
            "Adjust the number of dispatchers on duty from the 7-day workload forecast",
            "Plan reinforcements for major event periods (Geneva festivals, demonstrations)",
            "Account for seasonal absenteeism rates (winter epidemics among staff)",
        ],
    },
    "hospital-capacity-forecasting": {
        "title": "Hospital Capacity Forecasting",
        "description": "Time-series predictive models anticipating emergency department saturation and the occupancy of intensive care, intermediate care and conventional (downstream) beds, enabling proactive patient-flow management.",
        "recommended_actions": [
            "Predict emergency department inflow and bed occupancy at 24 h and 48 h (predictive NEDOCS score)",
            "Coordinate in real time inpatient discharges and transfers to rehabilitation and follow-up care units",
            "Trigger automatic alerts and cross-border bed-management crisis cells when capacity is strained",
            "Model the impact of emergency department overcrowding on ambulance turnaround and diversion times",
        ],
    },
    "demand-forecasting": {
        "title": "EMS Demand Forecasting",
        "description": "Hybrid forecasting models (Prophet, LightGBM, LSTM) integrating weather data, the calendar, school holidays and epidemiological surveillance to estimate the volume of emergency calls precisely and size the teams.",
        "recommended_actions": [
            "Integrate local weather (Open-Meteo) and epidemic (Sentinelles network) feeds in real time to refine forecasts",
            "Visualise the hourly EMS demand forecast per geographic sector over a D+1 to D+7 horizon",
            "Automatically alert operational managers when the actual call volume deviates significantly (> 15%) from the baseline forecast",
            "Use demand forecasts to dynamically adapt duty planning and the number of operational vehicles",
        ],
    },
    "resource-allocation": {
        "title": "Optimised Resource Allocation",
        "description": "Distributing human and material resources so as to maximise the effectiveness of the emergency response.",
        "recommended_actions": [
            "Allocate mobile ICU ambulances first to life-threatening emergencies",
            "Optimise the distribution of emergency equipment stocks across sites",
            "Track the activity status of every crew in real time",
        ],
    },
    "epidemic-early-warning": {
        "title": "Epidemic Early Warning",
        "description": "Early detection of weak epidemic signals from medical dispatch call patterns.",
        "recommended_actions": [
            "Monitor the trend of calls for influenza-like illness, gastroenteritis or respiratory distress",
            "Trigger an alert when a statistical incidence threshold is exceeded in a district",
            "Share early warnings with health authorities (FOPH, ARS) for coordinated action",
        ],
    },
    "surveillance": {
        "title": "Active Syndromic Surveillance",
        "description": "Continuous monitoring of population health indicators to identify anomalies or unusual clusters.",
        "recommended_actions": [
            "Analyse emergency visit data (SOS Médecins, hospitals) in real time",
            "Geographically identify abnormal clusters of cases with similar symptoms",
            "Adapt detection thresholds to seasonality and the local context",
        ],
    },
    "surge-management": {
        "title": "Surge Management",
        "description": "Operational strategies to cope with a sudden, massive rise in demand for emergency care.",
        "recommended_actions": [
            "Activate additional medical dispatch lines at the 15/144 centre",
            "Set up temporary reception structures (triage tents) in front of emergency departments",
            "Postpone non-urgent (scheduled) admissions to free capacity",
        ],
    },
    "pandemic-preparedness": {
        "title": "Pandemic Preparedness",
        "description": "Strategic planning and long-term modelling to strengthen the health system's resilience to global crises.",
        "recommended_actions": [
            "Establish business continuity plans for emergency and dispatch services",
            "Size strategic stocks of medical countermeasures (masks, antivirals, vaccines)",
            "Organise cross-border pandemic crisis simulation exercises",
        ],
    },
    "cross-border-coordination": {
        "title": "Cross-Border Health Coordination",
        "description": "Protocols and communication tools to harmonise the emergency response between France and Switzerland (Greater Geneva).",
        "recommended_actions": [
            "Interconnect the TECHWAN SAGA dispatch system (France) and its Swiss counterpart",
            "Establish free-passage agreements for rescue ambulances and helicopters",
            "Hold regular coordination meetings between the HUG, CHUV and SAMU management teams",
        ],
    },
    "situational-awareness": {
        "title": "Operational Situational Awareness",
        "description": "Real-time dashboard integrating every data source into a unified view of the emergency situation.",
        "recommended_actions": [
            "Display the real-time position of all mobile units (ambulances, mobile ICUs, helicopters)",
            "Integrate weather, epidemic and traffic feeds into a unified operational map",
            "Share the operational view with cross-border partners in real time",
        ],
    },
    "unassigned": {
        "title": "Unclassified Scenarios",
        "description": "Documents awaiting classification into a specific scenario.",
        "recommended_actions": [
            "Rerun the backfill script to reassign these documents",
            "Review titles and abstracts manually for manual classification",
        ],
    },
}

_LOCALIZED_KEYS = ("title", "description", "recommended_actions")


def localize_gesica(meta: dict[str, Any], lang: str | None) -> dict[str, Any]:
    """Return `meta` rendered in `lang`: a shallow copy with the English title,
    description and recommended actions when lang is English and the scenario is
    in the catalogue; the payload untouched otherwise (French is the stored text).
    `name` follows `title` so list cards and headers agree. Actions edited in the
    database (different count from the catalogue) are left as they are."""
    if not isinstance(meta, dict) or not (isinstance(lang, str) and lang.strip().lower().startswith("en")):
        return meta
    en = GESICA_EN.get(str(meta.get("id") or ""))
    if not en:
        return meta
    out = dict(meta)
    for key in _LOCALIZED_KEYS:
        if key not in en:
            continue
        if key == "recommended_actions":
            current = meta.get("recommended_actions")
            if isinstance(current, list) and current and len(current) != len(en[key]):
                continue                                   # edited on the server: keep as is
        out[key] = en[key]
    if "title" in en:
        out["name"] = en["title"]
        if isinstance(meta.get("label_short"), str):
            out["label_short"] = en["title"]
        if isinstance(meta.get("labelShort"), str):
            out["labelShort"] = en["title"]
    return out
