# TikTok Claims Classification Project 📱🧠

## Overview
This project was developed by the TikTok Data Team to create a machine learning model that classifies TikTok user-submitted videos as **claims** or **opinions**. Videos that make claims are more likely to violate platform guidelines and thus need to be prioritized for human moderation.

## Problem Statement
TikTok receives an overwhelming number of user reports on video content. Not all can be reviewed manually. The goal is to automate the detection of videos making factual claims (vs. personal opinions) to prioritize content moderation effectively.

## Dataset
The dataset contains various metadata about TikTok videos, including:
- `claim_status`: Indicates whether the video is a claim or an opinion.
- `video_view_count`, `like_count`, `comment_count`, `share_count`, `download_count`: Engagement metrics.
- `video_duration`: Length of the video in seconds.
- `verified_status`: Whether the user is verified.

## Exploratory Data Analysis (EDA)
- Data was cleaned and visualized to identify trends in engagement.
- Found that **view count**, **like count**, and **comment count** are heavily right-skewed.
- Verified accounts post more **opinions**, while unverified accounts post more **claims**.
- Over 200 null values were identified across 7 columns.

## Statistical Testing
A two-sample hypothesis test showed a significant difference in `video_view_count` between verified and unverified accounts.

## Regression Analysis
A logistic regression model was built to predict `verified_status`. 
- **F1 Score:** 66%
- Key Insight: **Longer videos** are more likely to be posted by verified users.

## Machine Learning Models
Two classification models were trained:
- **Random Forest**
- **XGBoost**

📌 **Final Model:** Random Forest  
📈 **Recall Score:** 0.995 (selected as the best model)  
🔍 Only 5 videos misclassified out of 3,817 in the test dataset.

### Key Features for Prediction
- `video_view_count`
- `like_count`
- `share_count`
- `download_count`

These engagement metrics provided the strongest predictive power for determining whether a video is a **claim** or an **opinion**.

## Conclusion
The final model is highly performant and can be used to automatically classify TikTok videos for moderation. It allows prioritizing videos that are more likely to violate the platform’s policies.

---


