"""
=========================================================
Grant Proposal Dataset Generator
Generates 210 Grant Proposal Summaries
Author : Your Name
=========================================================
"""

import random
import pandas as pd

random.seed(42)

domains = [
    "Healthcare",
    "Agriculture",
    "Education",
    "Climate Science",
    "Finance",
    "Cybersecurity",
    "Robotics",
    "Manufacturing",
    "Transportation",
    "Energy",
    "Smart Cities",
    "Environmental Science",
    "Social Media",
    "Retail"
]

methods = [
    "Deep Learning",
    "Machine Learning",
    "Graph Neural Networks",
    "Transformer Models",
    "Computer Vision",
    "Natural Language Processing",
    "Reinforcement Learning",
    "Federated Learning",
    "Blockchain",
    "Internet of Things",
    "Edge Computing",
    "Big Data Analytics",
    "Explainable AI",
    "Predictive Analytics"
]

funding = [
    "National Science Foundation",
    "Department of Biotechnology",
    "AI Research Council",
    "Innovation Fund",
    "Government Research Board",
    "Industrial Research Council",
    "International Research Foundation"
]

objectives = [
    "improve prediction accuracy",
    "enhance decision making",
    "reduce operational costs",
    "automate monitoring",
    "support early detection",
    "improve resource utilization",
    "increase sustainability",
    "optimize workflows",
    "provide intelligent recommendations",
    "strengthen security"
]

datasets = [
    "sensor data",
    "satellite images",
    "medical records",
    "financial transactions",
    "text documents",
    "IoT streams",
    "historical databases",
    "social media posts",
    "clinical reports",
    "industrial logs"
]

titles = [
    "Intelligent",
    "AI-Based",
    "Smart",
    "Advanced",
    "Automated",
    "Scalable",
    "Robust",
    "Novel",
    "Hybrid",
    "Efficient"
]

applications = {
    "Healthcare": [
        "Disease Diagnosis",
        "Cancer Detection",
        "Medical Imaging",
        "Patient Monitoring",
        "Drug Discovery"
    ],
    "Agriculture": [
        "Crop Disease Detection",
        "Yield Prediction",
        "Soil Monitoring",
        "Smart Irrigation",
        "Livestock Health"
    ],
    "Education": [
        "Student Performance Prediction",
        "Personalized Learning",
        "Curriculum Recommendation",
        "Attendance Analysis",
        "Learning Analytics"
    ],
    "Climate Science": [
        "Weather Prediction",
        "Flood Forecasting",
        "Climate Modeling",
        "Carbon Monitoring",
        "Air Quality Analysis"
    ],
    "Finance": [
        "Fraud Detection",
        "Risk Assessment",
        "Credit Scoring",
        "Stock Prediction",
        "Customer Analytics"
    ],
    "Cybersecurity": [
        "Intrusion Detection",
        "Phishing Detection",
        "Malware Analysis",
        "Threat Intelligence",
        "Network Security"
    ],
    "Robotics": [
        "Robot Navigation",
        "Autonomous Drones",
        "Industrial Robots",
        "Warehouse Automation",
        "Human Robot Collaboration"
    ],
    "Manufacturing": [
        "Predictive Maintenance",
        "Quality Inspection",
        "Production Planning",
        "Supply Chain Optimization",
        "Fault Detection"
    ],
    "Transportation": [
        "Traffic Prediction",
        "Autonomous Vehicles",
        "Route Optimization",
        "Accident Detection",
        "Fleet Management"
    ],
    "Energy": [
        "Renewable Energy Forecasting",
        "Smart Grid",
        "Power Consumption Prediction",
        "Solar Optimization",
        "Wind Energy Analysis"
    ],
    "Smart Cities": [
        "Waste Management",
        "Urban Planning",
        "Traffic Monitoring",
        "Public Safety",
        "Energy Efficiency"
    ],
    "Environmental Science": [
        "Wildlife Monitoring",
        "Water Quality",
        "Forest Monitoring",
        "Pollution Detection",
        "Disaster Management"
    ],
    "Social Media": [
        "Fake News Detection",
        "Sentiment Analysis",
        "Community Detection",
        "Recommendation Systems",
        "Trend Analysis"
    ],
    "Retail": [
        "Customer Recommendation",
        "Demand Forecasting",
        "Inventory Prediction",
        "Sales Analytics",
        "Market Basket Analysis"
    ]
}

records = []

proposal_id = 1

# ----------------------------------------------------------
# Generate 190 normal proposals
# ----------------------------------------------------------

for _ in range(190):

    domain = random.choice(domains)

    method = random.choice(methods)

    application = random.choice(applications[domain])

    title = random.choice(titles)

    objective = random.choice(objectives)

    dataset = random.choice(datasets)

    proposal_title = f"{title} {application} using {method}"

    summary = (
        f"This project proposes the use of {method} techniques for "
        f"{application.lower()} in the {domain.lower()} domain. "
        f"The study aims to {objective} using {dataset}. "
        f"The proposed framework includes data preprocessing, "
        f"model development, evaluation, and deployment to improve "
        f"research outcomes and practical applications."
    )

    records.append([
        f"P{proposal_id:03}",
        proposal_title,
        summary,
        domain,
        method,
        random.choice(funding)
    ])

    proposal_id += 1

# ----------------------------------------------------------
# Generate 20 Special-Case Proposal Pairs (40 proposals)
# Same methodology, different domain
# ----------------------------------------------------------

special_method = [
    "Graph Neural Networks",
    "Transformer Models",
    "Deep Learning",
    "Computer Vision",
    "Federated Learning",
    "Machine Learning",
    "Natural Language Processing",
    "Blockchain",
    "Predictive Analytics",
    "Explainable AI"
]

for method in special_method:

    d1, d2 = random.sample(domains, 2)

    app1 = random.choice(applications[d1])

    app2 = random.choice(applications[d2])

    for domain, app in [(d1, app1), (d2, app2)]:

        title = f"{app} using {method}"

        summary = (
            f"This proposal develops a {method} framework for "
            f"{app.lower()} within the {domain.lower()} domain. "
            f"The research focuses on improving performance through "
            f"advanced learning algorithms, optimization strategies, "
            f"and intelligent decision support systems."
        )

        records.append([
            f"P{proposal_id:03}",
            title,
            summary,
            domain,
            method,
            random.choice(funding)
        ])

        proposal_id += 1

# ----------------------------------------------------------
# Create DataFrame
# ----------------------------------------------------------

columns = [
    "Proposal_ID",
    "Title",
    "Summary",
    "Domain",
    "Methodology",
    "Funding_Agency"
]

df = pd.DataFrame(records, columns=columns)

# Shuffle

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save CSV

filename = "grant_proposals_dataset.csv"

df.to_csv(filename, index=False)

print("=" * 55)
print("Dataset Generated Successfully")
print(f"Total Records : {len(df)}")
print(f"Saved as      : {filename}")
print("=" * 55)

print(df.head())