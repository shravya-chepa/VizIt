import streamlit as st
import pandas as pd
from mitosheet.streamlit.v1 import spreadsheet
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="VizIt",
    page_icon="📈",
)



st.markdown("<h1 style='text-align: center; color: #D04848; font-size: 60px'>VizIt</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(label="Choose a CSV file", type=["csv"])


df = None
numeric_columns = []
categorical_columns = []

if uploaded_file is not None:
    try: 
        df = pd.read_csv(uploaded_file)
        st.data_editor(df)
    
        numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
        categorical_columns = [None] + list(df.select_dtypes(exclude='number').columns)
    except Exception as e:
        print(e)
        

with st.expander("Bar Plot"):
    col3, col4 = st.columns([0.25, 0.75])
    with col3:
        x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=3)
        y_values = st.selectbox('Y axis:', options=numeric_columns, index=1, key=4)
        c_axis = st.selectbox('Color: ', options=categorical_columns, key=34-1)
        
    with col4:
        fig, ax = plt.subplots()
        sns.barplot(x=x_values, y=y_values, data=df, palette='pastel', hue=c_axis)
        st.pyplot(fig)

with st.expander("Line Plot"):
    col15, col16 = st.columns([0.25, 0.75])
    with col15:
        x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=15)
        y_values = st.selectbox('Y axis:', options=numeric_columns, index=1, key=16)
        c_axis = st.selectbox('Color: ', options=categorical_columns, key=1516-1)
        style = st.selectbox('Style: ', options=categorical_columns, key=1516-2)
    with col16:
        fig, ax = plt.subplots()
        sns.lineplot(x=x_values, y=y_values, data=df, hue=c_axis, style=style, palette='pastel')
        st.pyplot(fig)

with st.expander("Scatter Plot"):
    col1 , col2 = st.columns([0.25, 0.75])
    with col1:
        x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=1)
        y_values = st.selectbox('Y axis:', options=numeric_columns, index=1, key=2)
        c_axis = st.selectbox('Color: ', options=categorical_columns)
        style = st.selectbox('Style: ', options=categorical_columns)
    with col2:
        fig, ax = plt.subplots()
        sns.scatterplot(x=x_values, y=y_values, data=df, hue=c_axis, palette='pastel', style=style)
        st.pyplot(fig)

with st.expander("Histogram"):
    col23 , col24 = st.columns([0.25, 0.75])
    with col23:
        x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=23)
        kde = st.selectbox('KDE: ', options=[True, False], key=2324-1)
        c_axis = st.selectbox('Color: ', options=categorical_columns, key=2324-2)
    with col24:
        fig, ax = plt.subplots()
        sns.histplot(x=x_values, data=df, kde=kde, hue=c_axis, palette='pastel')
        st.pyplot(fig)

with st.expander("Pie Chart"):
    col21, col22 = st.columns([0.25, 0.75])
    pastel_colors = sns.color_palette('pastel')
    with col21:
        pie_column = st.selectbox('Select a column for Pie chart:', options=categorical_columns)
    with col22:
        try:
            fig, ax = plt.subplots()
            plt.pie(df[pie_column].value_counts(), labels=df[pie_column].unique(), autopct='%1.1f%%', colors=pastel_colors)
            st.pyplot(fig)
        except:
            st.write("")

with st.expander("Box Plot"):
    col7, col8 = st.columns([0.25, 0.75])
    with col7:
        x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=7)
        y_values = st.selectbox('Y axis:', options=numeric_columns, index=1, key=8)
        c_axis = st.selectbox('Color: ', options=categorical_columns, key=78-1)
        orient = st.selectbox('Orientation: ', options=['v', 'h'], key=78-2)
    with col8:
        fig, ax = plt.subplots()
        sns.boxplot(x=x_values, y=y_values, data=df, palette='pastel', hue=c_axis, orient=orient)
        st.pyplot(fig)

with st.expander("Point Plot"):
    col11, col12 = st.columns([0.25, 0.75])
    with col11:
        x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=11)
        y_values = st.selectbox('Y axis:', options=numeric_columns, index=1, key=12)
        c_axis = st.selectbox('Color: ', options=categorical_columns, key=1112-1)
    with col12:
        fig, ax = plt.subplots()
        sns.pointplot(x=x_values, y=y_values, data=df, hue=c_axis, palette='pastel')
        st.pyplot(fig)


with st.expander("Density Plot"):
    col25 , col26 = st.columns([0.25, 0.75])
    with col25:
        x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=25)
        shade = st.selectbox('Shade: ', options=[True, False], key=26)
    with col26:
        try:
            fig, ax = plt.subplots()
            sns.kdeplot(df[x_values], shade=shade, palette='pastel')
            st.pyplot(fig)
        except:
            st.write("")


with st.expander("Swarm Plot"):
    col5, col6 = st.columns([0.25, 0.75])
    with col5:
        x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=5)
        y_values = st.selectbox('Y axis:', options=numeric_columns, index=1, key=6)
        c_axis = st.selectbox('Color: ', options=categorical_columns, key=56-1)
        orient = st.selectbox('Orientation: ', options=['v', 'h'], key=56-2)
    with col6:
        fig, ax = plt.subplots()
        sns.swarmplot(x=x_values, y=y_values, data=df, hue=c_axis, orient=orient, palette='pastel')
        st.pyplot(fig)



with st.expander("Violin Plot"):
    col9, col10 = st.columns([0.25, 0.75])
    with col9:
        x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=9)
        y_values = st.selectbox('Y axis:', options=numeric_columns, index=1, key=10)
        c_axis = st.selectbox('Color: ', options=categorical_columns, key=910-1)
        orient = st.selectbox('Orientation: ', options=['v', 'h'], key=910-2)
    with col10:
        fig, ax = plt.subplots()
        sns.violinplot(x=x_values, y=y_values, data=df, hue=c_axis, orient=orient, palette='pastel')
        st.pyplot(fig)


with st.expander("KDE Plot"):
    col13, col14 = st.columns([0.25, 0.75])
    with col13:
        x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=13)
        y_values = st.selectbox('Y axis:', options=numeric_columns, index=1, key=14)
        c_axis = st.selectbox('Color: ', options=categorical_columns, key=1314-1)
        fill = st.selectbox('Fill: ', options=[True, False], key=1314-2)
    with col14:
        fig, ax = plt.subplots()
        sns.kdeplot(x=x_values, y=y_values, data=df, hue=c_axis, fill=fill, palette='pastel')
        st.pyplot(fig)



with st.expander("Correlation Plot"):
    col17, col18 = st.columns([0.25, 0.75])
    with col17:
        x_values = st.selectbox('X axis:', options=[None], index=0, key=17)
        y_values = st.selectbox('Y axis:', options=[None], index=0, key=18)
    with col18:
        try:
            fig, ax = plt.subplots()
            sns.heatmap(df.select_dtypes(include='number').corr(),
                cmap=sns.cubehelix_palette(20, light=0.95, dark=0.45))
            st.pyplot(fig)
        except:
            st.write("")

with st.expander("Heat Map"):
    col19, col20 = st.columns([0.25, 0.75])
    with col19:
        x_values = st.selectbox('X axis:', options=[None], index=0, key=19)
        y_values = st.selectbox('Y axis:', options=[None], index=0, key=20)
    with col20:
        try:
            fig, ax = plt.subplots()
            sns.heatmap(df.select_dtypes(['float', 'int']), cmap=sns.cubehelix_palette(20, light=0.69, dark=0.25))
            st.pyplot(fig)
        except:
            st.write("")



