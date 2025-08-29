# Machine Learning Final Project  

This is the final project of our Machine Learning bootcamp, where we demonstrate the skills and knowledge acquired throughout our studies. Throughout this bootcamp, we have studied different models based on projects of different areas and types. Now it's time to create our own project using the algorithm that we think is best suited to our problem.  

We will have to find a suitable dataset to work with, process it, train a model and finally make it available for consumption.  

> *"Hard work always beats talent when talent doesn't work hard"* - Tim Notke  

---

## 👥 Credits  

**Team Members:**  
> - José Vicente Vareles Martínez

**Academy:**  
> - [4Geeks Academy](https://4geeksacademy.com/us/index)  
> - **Bootcamp:** Spain-DS-17  
> - **Mentor:** [Ing. Héctor Chocobar Torrejón](https://github.com/hchocobar/)  
> - **Teacher Assistant:** [Beatriz Solana Ros](https://github.com/mezcolantriz)  

---

## 🎯 Project Goal  

The goal of this project is to develop a complete end-to-end Machine Learning solution that includes:  
- Data acquisition and processing  
- Exploratory Data Analysis (EDA)  
- Model development and optimization  
- Web application deployment  
- Real-world problem-solving through ML techniques  

---

## 🚀 Project Overview  
This synthetic dataset simulates over 2 million detailed motor vehicle service and towing records from
2020 to 2024. This dataset provides realistic variability, enabling experiments in classification,
regression, clustering, and cost prediction.  

### Problem Statement  
We aim to analyze and predict **motor vehicle repair and towing operations**. This includes service 
duration, costs, customer satisfaction, and likelihood of towing vs repair services.  

### Dataset  
We use the **Motor Vehicle Repair & Towing Dataset**, which simulates over **2 million detailed service and towing records from 2020–2024**.  

It includes:  
- Service types (routine maintenance, emergency repairs, towing services, etc.)  
- Repair details (parts replaced, labor hours, diagnostics)  
- Customer demographics, loyalty, and satisfaction scores  
- Mechanic information (experience, specialization)  
- Towing specifics (distance, conditions, reasons)  
- Estimated vs. actual costs  
- Customer ratings and feedback  

### Methodology  
- Perform **data cleaning** and preprocessing  
- Conduct **exploratory analysis** to understand service/cost patterns  
- Develop **ML models** (regression for cost prediction, classification for service type prediction,
  clustering for customer segmentation)  
- Evaluate models with standard metrics  
- Deploy best model in a **Streamlit web app**  

### Results  
*[To be updated with our findings and model performance]*  

---

## 📝 Project Phases  

1. **Problem Definition**  
   Define the business and ML problem: predicting costs, satisfaction, or service outcomes.  

2. **Data Acquisition & Loading**  
   Load the provided dataset (`CSV`), explore structure, and preprocess.  

3. **Data Storage**  
   Optionally store the dataset in SQL for structured queries and scalability.  

4. **Descriptive Analysis**  
   Compute summary statistics, distributions, and correlations of service costs, duration, and feedback.  

5. **Full EDA**  
   Identify relevant features, visualize service/towing trends, handle missing values, and engineer new features.  

6. **Model Building & Optimization**  
   Train multiple models (Linear Regression, Random Forest, XGBoost, Decision Trees).  
   Perform hyperparameter tuning to optimize accuracy and interpretability.  

7. **Deployment**  
   Build a **Streamlit web app** to demonstrate the model predictions (e.g., predicting service cost or customer satisfaction).  
   Deploy on **Heroku/Render** for public access.  

---

## 📁 Project Structure  

```
ml-project-repo/
├── 📁 data/                # Raw and processed datasets
│    ├── 📁 interim/        # Intermediate transformed data
│    ├── 📁 processed/      # Final data for modeling
│    ├── 📁 raw/            # Original CSV files
├── 📁 database/            # SQL scripts and configs
├── 📁 docs/                # Documentation and presentations
├── 📁 models/              # Trained model artifacts
├── 📁 notebooks/           # Jupyter notebooks (EDA, analysis)
├── 📁 src/                 # Source code modules
├── 📁 webapp/              # Streamlit/Flask application
```  

---

## 🛠️ Technologies Used  

- **Data Processing:** Python (Pandas, NumPy), SQL  
- **EDA & Visualization:** Matplotlib, Seaborn, Plotly  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Deployment:** Streamlit, Heroku/Render  

---

## 📊 Results  

*[To be updated with metrics and insights]*  

---

## 🌐 Live Demo  

*[Link will be added after deployment]*  
