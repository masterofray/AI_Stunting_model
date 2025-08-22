# **An Interpretable AI Approach for Classifying Stunting in Mamberamo Raya, Papua, Indonesia**

**Author:** Aryanto

**Data Source:** Ministry of Health, Indonesia

### **Abstract**

Stunting remains a critical public health issue in Indonesia, particularly in remote regions like Mamberamo Raya, Papua. Early and accurate identification of children at risk is essential for targeted interventions. This study develops a machine learning model to classify stunting ("Stunting" vs. "Not Stunting") using demographic and environmental data. We employed an Extreme Gradient Boosting (XGBoost) classifier, a powerful and efficient algorithm. To ensure the model is not a "black box," we integrated interpretable Artificial Intelligence (AI) techniques, specifically SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations), to understand the key factors driving the predictions. After hyperparameter optimization, our XGBoost model achieved an accuracy of 88.67%, demonstrating high reliability. The SHAP analysis revealed that **Malnutrition Level** and **Difficult Access** to services are the most significant predictors of stunting in this region. This research highlights the potential of interpretable AI to provide actionable insights for public health officials to create more effective and targeted policies to combat stunting.

### **1\. Introduction**

Stunting, or impaired growth and development in children due to poor nutrition, repeated infection, and inadequate psychosocial stimulation, is a major challenge in Indonesia. The province of Papua, particularly the Mamberamo Raya regency, faces unique geographical and socio-economic challenges that contribute to a high prevalence of stunting. Addressing this issue requires a data-driven approach to identify the most vulnerable populations and the primary contributing factors.

Traditional statistical methods can identify correlations, but modern machine learning models often provide superior predictive power. However, the complexity of these models can make their decision-making processes opaque. This study aims to bridge this gap by building a highly accurate XGBoost classification model and leveraging interpretable AI to explain its predictions. The goal is to create a tool that not only predicts stunting risk but also provides clear, understandable reasons for its conclusions, empowering policymakers to design targeted and effective interventions. The data for this research was provided by the Ministry of Health of Indonesia.

### **2\. Methodology**

#### **2.1. Dataset**

The dataset consists of 607 records from Mamberamo Raya. The target variable is binary: **Stunting** (1) or **Not Stunting** (0). The following eight features (regressors) were used for prediction:

* **Malnutrition Level:** A numerical score indicating the severity of malnutrition.  
* **Difficult Access:** A binary indicator of whether the family has difficulty accessing healthcare and other essential services.  
* **District Code:** A categorical variable representing the specific district within the regency.  
* **Exclusive Milky:** A binary indicator for exclusive breastfeeding.  
* **Pure Water Access:** A binary indicator for access to clean drinking water.  
* **Gender:** A binary variable for the child's gender.  
* **Smoke Habit:** A binary indicator of smoking in the household.  
* **Healthy Toilet:** A binary indicator for access to sanitary toilet facilities.

The data was split into a training set (485 records) and a testing set (122 records) to build and evaluate the model.

#### **2.2. Modeling and Optimization**

An **XGBoost (Extreme Gradient Boosting)** classifier was chosen for its high performance and efficiency with tabular data. The model was initially trained with baseline parameters and subsequently optimized using the Optuna framework, a hyperparameter optimization tool. The optimization process aimed to maximize the F1-score, which is a robust metric for imbalanced datasets, over 80 trials. This resulted in a finely-tuned model with superior predictive capabilities.

#### **2.3. Interpretability**

To understand the model's behavior, we employed two state-of-the-art interpretable AI techniques:

* **SHAP (SHapley Additive exPlanations):** This technique assigns an "importance" value to each feature for every individual prediction. It provides a global understanding of which features are most influential across the entire dataset and shows how the value of a feature affects the prediction.  
* **LIME (Local Interpretable Model-agnostic Explanations):** LIME explains individual predictions by creating a simpler, interpretable model (like linear regression) that approximates the behavior of the complex XGBoost model in the vicinity of the prediction being explained.

### **3\. Results and Discussion**

#### **3.1. Model Performance**

After hyperparameter tuning, the final XGBoost model demonstrated excellent performance on the unseen test data. The overall accuracy reached **88.67%**. The classification report shows strong precision and recall for both classes, indicating that the model is effective at identifying both "Stunting" and "Not Stunting" cases correctly.

The confusion matrix below shows the specific prediction outcomes:

* **True Positives (Stunting):** 51  
* **True Negatives (Not Stunting):** 55  
* **False Positives (Predicted Stunting, was Not Stunting):** 2  
* **False Negatives (Predicted Not Stunting, was Stunting):** 14

The model is highly reliable in identifying children who are not stunted, and while there are some false negatives, its ability to correctly identify the majority of stunted children is strong.

#### **3.2. Feature Importance and Interpretation**

The SHAP analysis provided deep insights into the factors driving the model's predictions. The summary plot below illustrates the global feature importance. Features are ranked by their overall impact, and the plot shows how high (red) or low (blue) values of a feature contribute to the prediction.

The key findings from the SHAP analysis are:

1. **Malnutrition Level:** This is unequivocally the most important predictor. Higher malnutrition levels strongly push the model's prediction towards "Stunting." This confirms the direct biological link between nutritional status and physical growth.  
2. **Difficult Access:** This was the second most significant factor. The analysis shows that having difficult access to services is a strong indicator of stunting risk. This highlights the critical role of public infrastructure and healthcare accessibility.  
3. **Other Factors:** Features like District Code, Exclusive Milky, Smoke Habit, and Pure Water Access also contributed to the model's predictions, but with less impact than the top two. For example, the lack of exclusive breastfeeding (Exclusive\_Milky \= 0\) was associated with a higher risk of stunting.

LIME explanations for individual cases further validated these findings, showing how the combination of these factors led to a specific prediction for a given child.

### Strategy of Malnutrition Issue
Here are 3-5 actionable, community-focused intervention strategies to combat child stunting in rural Papua, Indonesia, focused on improving nutrition levels in low-resource settings:

* **Community-Led Nutrition Education and Cooking Demonstrations:** Train local *Kader Kesehatan* (health volunteers) and community leaders to conduct regular, interactive sessions on balanced diets using locally available ingredients, proper infant and young child feeding (IYCF) practices (e.g., exclusive breastfeeding, appropriate complementary feeding), and basic food hygiene. These sessions should include practical cooking demonstrations to show how to prepare nutritious meals with local resources.

* **Establishment of Community Food Gardens and Diversification:** Support households and communities in setting up small, sustainable vegetable and fruit gardens (*Pekarangan Pangan Lestari* concept). Promote the cultivation of nutrient-rich, resilient local crops (e.g., iron-rich leafy greens, vitamin A-rich root vegetables, local pulses, and fruits). Provide basic gardening training and initial seed/seedling support to enhance food diversity and access.

* **Mother-to-Mother Support Groups for IYCF and Hygiene:** Facilitate regular peer support groups for mothers (and engaging fathers where appropriate) led by trained community volunteers. These groups provide a safe space to discuss and share best practices on exclusive breastfeeding, timely and appropriate complementary feeding, and critical hygiene practices such as handwashing with soap at key times and safe water handling.

* **Strengthening *Posyandu* (Community Health Posts) for Growth Monitoring and Micronutrient Programs:** Improve community mobilization and engagement with existing *Posyandu* services. Ensure consistent and regular child growth monitoring for early detection of malnutrition, facilitate the regular distribution of essential micronutrients (e.g., Vitamin A supplements, deworming medicine), and provide iron-folic acid for pregnant women, all accompanied by targeted nutrition counseling.


### Strategy of Access Issue
Here are 3-5 actionable, community-focused intervention strategies to combat child stunting in rural Papua, Indonesia, tailored for difficult access and low-resource settings:

* **Community Health Worker (CHW) Empowerment & Mobile Outreach:** Train and equip local community members (e.g., Posyandu cadres, village health volunteers) in basic growth monitoring (weight/height for age), nutrition counseling (Infant and Young Child Feeding - IYCF practices, diversified local diets), hygiene promotion, and identifying danger signs. These CHWs would conduct regular, scheduled visits to remote hamlets, bringing essential services and education directly to families who cannot easily access health facilities.
* **Local Food System Diversification & Nutrition Education:** Promote and support community and household gardens for nutrient-rich, locally adaptable crops (e.g., leafy greens, legumes, local fruits) and small-scale livestock (e.g., chickens, fish farming). Conduct practical cooking demonstrations using these locally available diverse foods, focusing on preparing balanced meals for pregnant women, infants, and young children, integrating these sessions into existing community gatherings.
* **Village-Level Water, Sanitation, and Hygiene (WASH) Improvement:** Implement community-led initiatives to improve access to and use of safe water sources and latrines (e.g., Community-Led Total Sanitation - CLTS approach). Train villagers in simple, low-cost water purification methods and promote critical handwashing practices with soap at key times (before food preparation, after defecation) through local hygiene champions and community campaigns.
* **Integrated Mother and Child Health (MCH) Outreach Camps:** Organize periodic (e.g., quarterly) comprehensive mobile health camps at central community points within remote areas. These camps would offer bundled services including growth monitoring, immunizations, antenatal/postnatal care, family planning, nutrition counseling, and provision of essential micronutrients (e.g., Vitamin A, iron-folate) for both mothers and children, thereby reducing the need for multiple, distant trips to health centers.

### **4\. Conclusion**

This study successfully developed a high-accuracy XGBoost model for classifying stunting in Mamberamo Raya, Papua, achieving an accuracy of 88.67%. By integrating interpretable AI techniques like SHAP and LIME, we moved beyond simple prediction to uncover the "why" behind the model's decisions.

The results clearly indicate that **malnutrition level** and **difficult access to services** are the primary drivers of stunting in this region. This finding provides a clear and actionable focus for public health interventions. Policies should prioritize not only direct nutritional programs but also initiatives to improve infrastructure, such as roads and communication, to ensure that all communities can access essential health and nutritional services.

This work serves as a model for how advanced analytics and interpretable AI can be applied to complex public health challenges, providing robust evidence for data-driven policymaking.