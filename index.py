import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



from animation import animation

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

pd.set_option('mode.use_inf_as_na', True)


st.set_page_config(
    page_title="VizIt",
    page_icon="📈",
    layout="wide"
)

# with st.sidebar:
#     selected_radio = st.radio(
#         "",
#         ("Data Description and Exploration", "Data Cleaning", "Data Visualization", "Linear Regression", "Logistic Regression")
#     )

# print("selected radio: ", selected_radio)


st.markdown("<h1 style='text-align: center; color: #D04848; font-size: 60px'>VizIt</h1>", unsafe_allow_html=True)



st.markdown(animation,unsafe_allow_html=True)


css_style = """
<style>
.st-emotion-cache-1erivf3 {
    background: #D04848;
}

h3 {
    color: rgba(255, 220, 90, 0.95);
}
</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

uploaded_file = st.file_uploader(label="Choose a CSV file", type=["csv"])




df = None
numeric_columns = []
categorical_columns = []

if uploaded_file is not None:
    try: 
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Editor")
        st.data_editor(df, width=1000)
    
        numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
        categorical_columns = [None] + list(df.select_dtypes(exclude='number').columns)
    except Exception as e:
        print(e)

try:
    
    if uploaded_file:

        st.markdown("<h2 style='text-align: center; color: #D04848;'>Data Exploration</h2>", unsafe_allow_html=True)

        st.header("1. Structure")
        st.subheader("How big is the file")
        # print("file is",  os.path.getsize("path/filename.csv") / 1e6, "MB")
        bytes_data = uploaded_file.getvalue()
        file_size_MB = round(len(bytes_data) / (1024 * 1024), 6)
        st.write(str(file_size_MB) + "MB")

        st.subheader("Data shape")
        st.write(df.shape)
        st.write("Number of rows: " + str(df.shape[0]))
        st.write("Number of columns: " + str(df.shape[1]))

        col21, col22 = st.columns([0.5, 0.5])
        with col21: 
            st.subheader("All columns")
            column_names = df.columns.tolist()
            st.write(column_names)
        with col22:
            st.subheader("Column types")
            st.dataframe(df.dtypes, width=300)
        
        
        

        st.subheader("Head of data")
        head_number = st.number_input("Enter number of lines to return in head" , min_value=1, max_value=20, value=5, step=1, key="head-1")
        st.write(df.head(head_number))

        st.subheader("Tail of data")
        tail_number = st.number_input("Enter number of lines to return in tail" , min_value=1, max_value=20, value=5, step=1, key="tail-1")
        st.write(df.tail(tail_number))

        st.subheader("Description")
        st.write("For numerical columns")
        st.write(df.describe())
        # describing categorical data
        st.write("For categorical columns")
        st.write(df.describe(include=['object']))

        st.header("2. Granularity")
        st.subheader("Unique values")

        unique_counts = df.nunique()
        st.write("Number of unique values in each column: ")
        st.dataframe(unique_counts, width=500)

        cat_col = st.selectbox('Choose a categorical column:', options=categorical_columns, index=0, key="unique-1")
        if cat_col:
            unique_values = df[cat_col].unique()
            st.write(unique_values)

        

        st.header("3. Scope")
        st.write("How incomplete is your data")
        

        

        col23, col24, col25 = st.columns([0.33, 0.33, 0.33])
        with col23:
            missing_values = df.isnull().sum()
            st.write("Missing values:")
            st.dataframe(missing_values, width=300)

        with col24:
            blank_values = (df == '').sum()
            st.write("Blank values:")
            st.dataframe(blank_values, width=300)

        with col25:
            nan_values = df.isna().sum()
            st.write("NaN values:")
            st.dataframe(nan_values, width=300)

        st.header("4. Temporality")
        st.write("Please choose a column that represents valid datetime")

        time_col_option = st.selectbox('Choose a categorical column:', options=df.columns, index=0, key="time-1")

        try:
            time_col = pd.to_datetime(df[time_col_option], errors="coerce")
            st.dataframe(time_col, width=500)
            min_time = time_col.min()
            max_time = time_col.max()

            # Display min and max times
            st.write(f"Min time represented in {time_col_option}: {min_time.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"Max time represented in {time_col_option}: {max_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        except Exception as e:
            print(e)
            st.write("The selected column cannot be changed to datetime type")

        st.markdown("<h2 style='text-align: center; color: #D04848;'>Data Cleaning</h2>", unsafe_allow_html=True)
        

        st.subheader("Drop rows or columns with missing values")
        drop_all = st.checkbox("Both rows and columns")
        drop_rows = st.checkbox("Rows")
        drop_columns = st.checkbox("Columns")

        if drop_all:
            df.dropna(inplace=True)

        if drop_rows:
            df.dropna(axis=0, inplace=True)

        if drop_columns:
            df.dropna(axis=1, inplace=True)

        st.write("NaN count after dropping")
        na_count_rows = df.isna().sum()
        st.dataframe(na_count_rows, width=600)

        st.subheader("Fill missing values with a specific value")
        fill_col_option_numeric = st.selectbox('Numeric columns', options=[None]+numeric_columns, index=0, key="fill-num-0")

        if fill_col_option_numeric != None:
            fill_options = st.radio("Select among the following",["none of the below", "0", "median", "mode"])
            if fill_options == "constant":
                fill_value = st.number_input("Enter constant value and press enter", value=0, step=1)
            elif fill_options == "Median":
                fill_value = df[fill_col_option_numeric].median()
            elif fill_options == "Mode":
                fill_value = df[fill_col_option_numeric].mode().iloc[0]
            elif fill_options == "0":
                fill_value = 0
            else:
                fill_value = None

            if fill_value:
                df[fill_col_option_numeric].fillna(fill_value, inplace=True)

        

        fill_col_option_categorical = st.selectbox('Categorical columns', options=categorical_columns, index=0, key="fill-num-1")

        if fill_col_option_categorical != None:
            fill_options_2 = st.radio("Select among the following",["none of the below", "Mode", "Unknown", "Custom"])
            if fill_options_2 == "Custom":
                fill_value = st.text_input("Enter text value and press enter")
            elif fill_options_2 == "Mode":
                fill_value = df[fill_col_option_categorical].mode().iloc[0]
            elif fill_options_2 == "Unknown":
                fill_value = "Unknown"
            else:
                fill_value = None

            if fill_value:
                df[fill_col_option_categorical].fillna(fill_value, inplace=True)
            

        st.write("NaN count after dropping")
        na_count_rows = df.isna().sum()
        st.dataframe(na_count_rows, width=600)

        st.subheader("Remove duplicates")
        duplicate_count = df.duplicated().sum()
        st.write("Number of duplicates in data: ", duplicate_count)
        duplicates = df[df.duplicated(keep=False)]
        st.dataframe(duplicates, width=700)
        remove_duplicates_options = st.radio("Select among the following", ["no", "yes"])
        if remove_duplicates_options == "yes":
            df.drop_duplicates(inplace=True)
            duplicate_count = df.duplicated().sum()
            st.write("Number of duplicates in data now: ", duplicate_count)
        
        



        
        st.markdown("<h2 style='text-align: center; color: #D04848;'>Data Visualization</h2>", unsafe_allow_html=True)


        with st.expander("Bar Plot"):
            col3, col4 = st.columns([0.25, 0.75])
            with col3:
                x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=3)
                y_values = st.selectbox('Y axis:', options=numeric_columns, index=1, key=4)
                c_axis = st.selectbox('Color: ', options=categorical_columns, key=34-1)

            with col4:
                fig, ax = plt.subplots()
                if c_axis is not None:
                    sns.barplot(x=x_values, y=y_values, data=df, palette='pastel', hue=c_axis)
                else:
                    sns.barplot(x=x_values, y=y_values, data=df)
                st.pyplot(fig)
                plt.close(fig)

        with st.expander("Line Plot"):
            col15, col16 = st.columns([0.25, 0.75])
            with col15:
                x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=15)
                y_values = st.selectbox('Y axis:', options=numeric_columns, index=1, key=16)
                c_axis = st.selectbox('Color: ', options=categorical_columns, key=1516-1)
                style = st.selectbox('Style: ', options=categorical_columns, key=1516-2)
            with col16:
                fig, ax = plt.subplots()
                if c_axis is not None:
                    sns.lineplot(x=x_values, y=y_values, data=df, hue=c_axis, style=style, palette='pastel')
                else:
                    sns.lineplot(x=x_values, y=y_values, data=df, style=style, )
                plt.close(fig)
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
                if c_axis is not None:
                    sns.scatterplot(x=x_values, y=y_values, data=df, hue=c_axis, palette='pastel', style=style)
                else:
                    sns.scatterplot(x=x_values, y=y_values, data=df, style=style)
                plt.close(fig)
                st.pyplot(fig)

        with st.expander("Histogram"):
            col23 , col24 = st.columns([0.25, 0.75])
            with col23:
                x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=23)
                kde = st.selectbox('KDE: ', options=[True, False], key=2324-1)
                c_axis = st.selectbox('Color: ', options=categorical_columns, key=2324-2)
            with col24:
                fig, ax = plt.subplots()
                if c_axis is not None:
                    sns.histplot(x=x_values, data=df, kde=kde, hue=c_axis, palette='pastel')
                else:
                    sns.histplot(x=x_values, data=df, kde=kde)

                plt.close(fig)
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
                    plt.close(fig)

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
                if c_axis is not None:
                    sns.boxplot(x=x_values, y=y_values, data=df, palette='pastel', hue=c_axis, orient=orient)
                else:
                    sns.boxplot(x=x_values, y=y_values, data=df, orient=orient)
                st.pyplot(fig)
                plt.close(fig)


        with st.expander("Point Plot"):
            col11, col12 = st.columns([0.25, 0.75])
            with col11:
                x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=11)
                y_values = st.selectbox('Y axis:', options=numeric_columns, index=1, key=12)
                c_axis = st.selectbox('Color: ', options=categorical_columns, key=1112-1)
            with col12:
                fig, ax = plt.subplots()
                if c_axis is not None:
                    sns.pointplot(x=x_values, y=y_values, data=df, hue=c_axis, palette='pastel')
                else:
                    sns.pointplot(x=x_values, y=y_values, data=df)
                st.pyplot(fig)
                plt.close(fig)



        with st.expander("Density Plot"):
            col25 , col26 = st.columns([0.25, 0.75])
            with col25:
                x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=25)
                shade = st.selectbox('Shade: ', options=[True, False], key=26)
            with col26:
                try:
                    fig, ax = plt.subplots()
                    sns.kdeplot(df[x_values], fill=shade)
                    st.pyplot(fig)
                    plt.close(fig)

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
                if c_axis is not None:
                    sns.swarmplot(x=x_values, y=y_values, data=df, hue=c_axis, orient=orient, palette='pastel')
                else:
                    sns.swarmplot(x=x_values, y=y_values, data=df, orient=orient)

                st.pyplot(fig)
                plt.close(fig)




        with st.expander("Violin Plot"):
            col9, col10 = st.columns([0.25, 0.75])
            with col9:
                x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=9)
                y_values = st.selectbox('Y axis:', options=numeric_columns, index=1, key=10)
                c_axis = st.selectbox('Color: ', options=categorical_columns, key=910-1)
                orient = st.selectbox('Orientation: ', options=['v', 'h'], key=910-2)
            with col10:
                fig, ax = plt.subplots()
                if c_axis is not None:
                    sns.violinplot(x=x_values, y=y_values, data=df, hue=c_axis, orient=orient, palette='pastel')
                else:
                    sns.violinplot(x=x_values, y=y_values, data=df, orient=orient)
                st.pyplot(fig)
                plt.close(fig)



        with st.expander("KDE Plot"):
            col13, col14 = st.columns([0.25, 0.75])
            with col13:
                x_values = st.selectbox('X axis:', options=numeric_columns, index=0, key=13)
                y_values = st.selectbox('Y axis:', options=numeric_columns, index=1, key=14)
                c_axis = st.selectbox('Color: ', options=categorical_columns, key=1314-1)
                fill = st.selectbox('Fill: ', options=[True, False], key=1314-2)
            with col14:
                fig, ax = plt.subplots()
                if c_axis is not None:
                    sns.kdeplot(x=x_values, y=y_values, data=df, hue=c_axis, fill=fill, palette='pastel')
                else:
                    sns.violinplot(x=x_values, y=y_values, data=df, orient=orient)
                st.pyplot(fig)
                plt.close(fig)




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
                    plt.close(fig)

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
                    plt.close(fig)

                except:
                    st.write("")

        st.markdown("<h2 style='text-align: center; color: #D04848;'>Simple Linear Regression</h2>", unsafe_allow_html=True)

        st.subheader("Correlation")
        st.write(df[numeric_columns].corr())
        x = st.selectbox('x ', options=[None]+numeric_columns, index=0, key="simple-linear-x")
        y = st.selectbox('Y (target): ', options=[None]+numeric_columns, index=0, key="simple-linear-y")

        if x != None and y != None:
                    df[x] = df[x] + np.random.randn(len(df))/2
                    df[y] = df[y] + np.random.randn(len(df))/2

                    st.subheader("Scatterplot of x and Y")

                    fig = px.scatter(df, x=x, y=y)
                    st.plotly_chart(fig)

                    predicted_y = y + '_predicted'

                    def predict_mean_y(actual):
                        return df.loc[np.abs(df[x] - actual) <= 0.5, y].mean()
                    
                    

                    st.subheader("Head of data")
                    df[predicted_y] = df[x].apply(predict_mean_y)

                    st.write(df.head())

                    st.subheader("Scatterplot of x, Y and Predicted Means")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='markers', name='actual'))
                    fig.add_trace(go.Scatter(x=df[x], y=df[predicted_y], mode='markers', name='predicted means', line=dict(color='gold')))
                    fig.update_layout(xaxis_title=x, yaxis_title=y)

                    st.plotly_chart(fig)

                    st.subheader("Implementing optimal coefficients and plotting linear model")
                    
                    def calculate_coefficients(X, Y):
                        x_mean = X.mean()
                        y_mean = Y.mean()
                        xy_mean = (X * Y).mean()
                        x_square_mean = (X ** 2).mean()
                        m = ((x_mean * y_mean) - xy_mean) / ((x_mean) ** 2 - x_square_mean)
                        b = y_mean - (m * x_mean)
                        return m, b
                            
                    X = df[x]
                    Y = df[y]

                    
                    m, b = calculate_coefficients(X, Y)

                    st.write("m= ", str(np.round(m, 2)))
                    st.write("b= ", str(np.round(b, 2)))

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='markers', name='actual'))
                    fig.add_trace(go.Scatter(x=df[x], y=df[predicted_y], mode='markers', name='predicted means', line=dict(color='gold')))
                    fig.add_trace(go.Scatter(x = df[x], y = m * df[x] + b, name = 'linear model', line=dict(color='red')))

                    fig.update_layout(xaxis_title=x, yaxis_title=y)

                    st.plotly_chart(fig)

                    st.subheader("Visualizing Loss Surface")
                    def mse(y, yhat):
                        return np.mean((y - yhat)**2)
                    
                    def mse_for_height_model(t):
                        a, b = t
                        return mse(df[x], a + b*df[y])
                    
                    num_points = 200 # increase for better resolution, but it will run more slowly. 

                    # if (num_points <= 100):

                    uvalues = np.linspace(20, 32, num_points)
                    vvalues = np.linspace(-1, 3, num_points)
                    (u,v) = np.meshgrid(uvalues, vvalues)
                    thetas = np.vstack((u.flatten(),v.flatten()))

                    MSE = np.array([mse_for_height_model(t) for t in thetas.T])

                    try:
                        loss_surface = go.Surface(x=u, y=v, z=np.reshape(MSE, u.shape))

                        opt_point = go.Scatter3d(x=[m], y=[b], z=[mse_for_height_model((m, b))],
                                                mode='markers', name='optimal parameters',
                                                marker=dict(size=1, color='gold'))

                        fig = go.Figure(data=[loss_surface])
                        fig.add_trace(opt_point)

                        fig.update_layout(scene=dict(
                            xaxis_title="theta0",
                            yaxis_title="theta1",
                            zaxis_title="MSE"))

                        # Display the plot on Streamlit
                        st.write("This plot is interactive!")
                        st.plotly_chart(fig)

                    except:
                        st.write("Plot too large to display")



        st.markdown("<h2 style='text-align: center; color: #D04848;'>Multinomial Logistic Regression</h2>", unsafe_allow_html=True)

        target_column = st.selectbox("Select target: ", options=categorical_columns)
        feature_columns = st.multiselect("Select features: ", options=numeric_columns)

        target_values = df[target_column].unique()
        st.subheader("Values in target ")
        st.write(target_values)


        col_log_1, col_log_2 = st.columns([0.65, 0.35])
        with col_log_1:
            st.subheader("Count plot for target")
            for value in target_values:
                count = df[df[target_column] == value].shape[0]
                st.write(f"{value}: {count}")
            
        with col_log_2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(x=target_column, data=df, ax=ax)
            st.pyplot(fig)
            plt.close(fig)


        
        target_dict = {val: idx for idx, val in enumerate(target_values)}
        target_column_name = target_column+"_num"
        df[target_column_name] = df[target_column].map(target_dict)

        st.subheader("Converting target values:")
        st.write(target_dict)

        X_log = df[feature_columns]
        y_log = df[target_column_name]
        X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.25, random_state=42)

        multinomial_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

        multinomial_model.fit(X_train, y_train)
        score = multinomial_model.score(X_test, y_test)

        st.subheader('New Instance Prediction')
        new_instance = {}
        for feature in feature_columns:
            new_instance[feature] = st.number_input(f'Enter value for {feature}:')

        new_instance_df = pd.DataFrame([new_instance])
        prediction = multinomial_model.predict(new_instance_df)
        predicted_category = list(target_dict.keys())[prediction[0]]

        st.markdown(f"<h3 style='color: #FFFFFF; font-size: 25px'>Predicted category - {predicted_category}</h3>", unsafe_allow_html=True)


        st.subheader("Metrics")
        score = multinomial_model.score(X_test, y_test)
        st.write(f'The accuracy of the model is: {score}')
        
        st.write("Classification report: ")
        y_pred = multinomial_model.predict(X_test)
        target_names = list(target_dict.keys())
        classification_rep = pd.DataFrame(classification_report(y_test, y_pred, target_names=target_names, output_dict=True)).T
        # classification_str = str(classification_rep)


        st.dataframe(classification_rep)









        
  
     
    else:
        st.write("Please upload a file first")
   

except Exception as e:
    print(e)




st.markdown("<div style='text-align: center; color: #D04848; font-size: 14px; margin-top: 50px'>©VizIt 2024</div>", unsafe_allow_html=True)


