import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset
def load_data():
    file_path = r"C:\Users\KIIT\Climate Minor Project\climate_debt_analyzer_top100_10yrs.xlsx"
    df = pd.read_excel(file_path)
    return df

df = load_data()

# Streamlit App
st.set_page_config(page_title='Climate Debt Analyzer', layout='wide')
st.title('🌍 Climate Debt Analyzer - 10 Years Analysis')

# Sidebar
st.sidebar.header('Filter Options')
selected_country = st.sidebar.selectbox('Select a Country', df['Country'].unique())
years = st.sidebar.slider('Select Year Range', int(df['Year'].min()), int(df['Year'].max()), (2015, 2024))

# Filtered Data
filtered_df = df[(df['Country'] == selected_country) & (df['Year'].between(years[0], years[1]))]

# Train a model to predict SDG Score
def train_model(data):
    features = ['Debt (Billion USD)', 'CO2 Emissions (Million Tons)', 'Renewable Energy Investment (Billion USD)']
    target = 'SDG Score'

    model_data = data.dropna(subset=features + [target])
    X = model_data[features]
    y = model_data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, rmse

model, model_rmse = train_model(df)

# Predict SDG Score for filtered data
if not filtered_df.empty:
    features = ['Debt (Billion USD)', 'CO2 Emissions (Million Tons)', 'Renewable Energy Investment (Billion USD)']
    filtered_df = filtered_df.dropna(subset=features)
    predicted_scores = model.predict(filtered_df[features])
    filtered_df['Predicted SDG Score'] = predicted_scores

# Debt and CO2 Emissions Over Time
col1, col2 = st.columns(2)
with col1:
    st.subheader(f'Debt Trend ({selected_country})')
    fig_debt = px.line(filtered_df, x='Year', y='Debt (Billion USD)', markers=True, title='Debt Over Time')
    st.plotly_chart(fig_debt, use_container_width=True)

with col2:
    st.subheader(f'CO₂ Emissions Trend ({selected_country})')
    fig_co2 = px.line(filtered_df, x='Year', y='CO2 Emissions (Million Tons)', markers=True, title='CO₂ Emissions Over Time')
    st.plotly_chart(fig_co2, use_container_width=True)

# Renewable Investment and SDG Score Comparison
col3, col4 = st.columns(2)
with col3:
    st.subheader(f'Renewable Energy Investment ({selected_country})')
    fig_invest = px.bar(filtered_df, x='Year', y='Renewable Energy Investment (Billion USD)', color='Year', title='Investment in Renewable Energy')
    st.plotly_chart(fig_invest, use_container_width=True)

with col4:
    st.subheader(f'SDG Score Trend ({selected_country})')
    fig_sdg = px.line(filtered_df, x='Year', y='SDG Score', markers=True, title='Sustainable Development Goal Score')
    st.plotly_chart(fig_sdg, use_container_width=True)

# Correlation Heatmap (Filtered)
st.subheader(f'📊 Correlation Insights ({selected_country})')
if filtered_df.shape[0] >= 2:
    corr_df = filtered_df[['Debt (Billion USD)', 'CO2 Emissions (Million Tons)', 'Renewable Energy Investment (Billion USD)', 'SDG Score']].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
else:
    st.info("Not enough data to show correlation heatmap. Please select a wider year range.")

# AI-Powered Policy Recommendations
st.subheader('📌 AI-Powered Policy Recommendations')
avg_sdg = filtered_df['SDG Score'].mean()
if avg_sdg < 65:
    st.warning(f"{selected_country} should significantly increase renewable energy investments and debt transparency.")
elif avg_sdg < 75:
    st.info(f"{selected_country} is making progress but should optimize climate debt usage for better impact.")
else:
    st.success(f"{selected_country} is on track with sustainable development goals. Keep up the good work!")

# Display ML Prediction Output
st.subheader('🧠 ML: Actual vs Predicted SDG Score')
st.write(f"Model RMSE: {model_rmse:.2f}")
st.dataframe(filtered_df[['Year', 'SDG Score', 'Predicted SDG Score']].round(2))

# Interactive Sliders to Simulate SDG Score
st.subheader("🎛 Simulate SDG Score Based on Custom Inputs")
with st.expander("Adjust Parameters to Simulate Outcome"):
    sim_debt = st.slider('Debt (Billion USD)', float(df['Debt (Billion USD)'].min()), float(df['Debt (Billion USD)'].max()), float(df['Debt (Billion USD)'].mean()))
    sim_co2 = st.slider('CO2 Emissions (Million Tons)', float(df['CO2 Emissions (Million Tons)'].min()), float(df['CO2 Emissions (Million Tons)'].max()), float(df['CO2 Emissions (Million Tons)'].mean()))
    sim_invest = st.slider('Renewable Energy Investment (Billion USD)', float(df['Renewable Energy Investment (Billion USD)'].min()), float(df['Renewable Energy Investment (Billion USD)'].max()), float(df['Renewable Energy Investment (Billion USD)'].mean()))

    sim_input = pd.DataFrame([[sim_debt, sim_co2, sim_invest]], columns=['Debt (Billion USD)', 'CO2 Emissions (Million Tons)', 'Renewable Energy Investment (Billion USD)'])
    sim_prediction = model.predict(sim_input)[0]

    st.success(f"🧠 Predicted SDG Score (Simulation): {sim_prediction:.2f}")

# Clustering Countries
def cluster_countries(data, n_clusters=3):
    cluster_features = ['Debt (Billion USD)', 'CO2 Emissions (Million Tons)', 'Renewable Energy Investment (Billion USD)', 'SDG Score']
    clustered = data.groupby('Country')[cluster_features].mean().dropna()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustered)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    clustered['Cluster'] = clusters
    clustered.reset_index(inplace=True)
    return clustered

clustered_df = cluster_countries(df)

# Visualize Clusters
st.subheader("🧬 Country Clusters Based on Climate & Development Patterns")
fig_cluster = px.scatter(clustered_df, x='Debt (Billion USD)', y='CO2 Emissions (Million Tons)',
                         color='Cluster', hover_name='Country',
                         size='Renewable Energy Investment (Billion USD)',
                         title='Clusters of Countries by Climate Metrics')
st.plotly_chart(fig_cluster, use_container_width=True)

# Dataset Download
st.sidebar.subheader('Download Dataset')
st.sidebar.download_button(label='📥 Download Data', data=df.to_csv(index=False), file_name='climate_debt_analyzer_10yrs.csv', mime='text/csv')