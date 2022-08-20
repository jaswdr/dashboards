from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import plotly.express as px
import streamlit as st

st.title("Titanic dataset exploration")

DATASET_FILE = './data/titanic.csv'
st.download_button("Download dataset", open(DATASET_FILE, 'r'))


@st.cache(suppress_st_warning=True)
def read_data():
    """Reads the local Titanic data set file"""
    with st.spinner("Loading data..."):
        return pd.read_csv(DATASET_FILE)


def comment(text, where=None):
    """Add personal comment."""
    write_to = st
    if where:
        write_to = where
    write_to.info(f'**Comment**: {text}')


df = read_data()

st.header("Exploration")

tab1, tab2, tab3, tab4 = st.tabs([
    'Samples', 'Columns Data Types', 'Data Types', 'Data Dictionary'])
with tab1:
    st.subheader('Sample')
    st.dataframe(df.sample(10))

with tab2:
    st.subheader("Columns data types")
    types = df.dtypes.to_dict()
    st.table(pd.DataFrame({
        'Column': [str(k) for k in types.keys()],
        'Type': [str(t) for t in types.values()]}))

with tab3:
    st.subheader('Data dictionary')
    data_dictionary = pd.DataFrame({
        'Feature': [
            'Survived', 'Pclass', 'Sex', 'Age',
            'SibSp', 'Parch', 'Ticket', 'Fare',
            'Cabin', 'Embarked'],
        'Definition': [
            'Survival', 'Ticket class', 'Sex',
            'Age in years', '# of siblings/spouses aboard',
            '# of parents/children aboard', 'Ticket number',
            'Passenger fare', 'Cabin number', 'Port of Embarkation'
        ],
        'Key': [
            '0 = No, 1 = Yes', '1 = 1st, 2 = 2nd, 3 = 3rd',
            '', '', '', '', '', '', '',
            'C = Cherbourg, Q = Queenstown, S = Southampton'
        ]
    })
    st.table(data_dictionary)

with tab4:
    st.subheader("Correlation Matrix")
    corr = df.corr()
    fig = px.imshow(corr)
    fig

    comment(
        "Pclass has strong negative correlation with `Survived`, "
        "and Fare has a not so strong positive correlation with it.")

st.subheader('Comparing each column to target')
denied = ['Survived', 'Name', 'Ticket', 'Cabin']
cols = sorted([c for c in df.columns.tolist() if c not in denied])
tabs = st.tabs(cols)
for index, col in enumerate(cols):
    tab = tabs[index]
    tab.subheader(f'Survivals by {col}')
    fig = px.histogram(
        df,
        x=col,
        color='Survived',
        text_auto=True)
    fig.update_layout(bargap=.2)
    tab.write(fig)

st.header("Cleanup")
st.subheader("Columns with missing data")


def missing_counter():
    data = (df.isnull() | df.empty | df.isna()).sum()
    data = data[data > 0].sort_values(ascending=False)
    return data


missing_count = missing_counter()
percent_missing = (missing_count/df.shape[0]).mul(100)
percent_missing.name = '% Missing'
st.table(percent_missing.apply(lambda x: '{:0.1f}%'.format(x)))

st.subheader("Dropping columns with >30% missing data")
with st.spinner("Dropping columns with >30% missing data"):
    to_drop = percent_missing[percent_missing > 30].index
    df = df.drop(to_drop, axis=1)
    st.write(f"Dropped columns: {to_drop.tolist()}")

st.subheader('Filling NA values')
st.write("Applying mean for NA values in Age column")
df['Age'] = df['Age'].fillna(df['Age'].mean())

st.write("Applying mode for NA values in Embarked column")
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode())

st.subheader('Adding new features')
st.write('Splitting Name in Surname and Gived name')
surnames = []
gived_names = []
for name in df['Name'].tolist():
    names = name.split(',')
    surnames.append(names[0])
    gived_names.append(' '.join(names[1:]))

df['Surname'] = surnames
df['Gived Name'] = gived_names
df = df.drop('Name', axis=1)

st.header("Preprocessing")
label_encoding_columns = ['Surname', 'Gived Name', 'Embarked', 'Ticket']
label_encoder = LabelEncoder()
for col in label_encoding_columns:
    df[col] = label_encoder.fit_transform(df[col].astype(str))
st.write(f'Applied LabelEncode for {label_encoding_columns}')

onehot_encoding_columns = ['Sex']
onehot_encoder = OneHotEncoder()
for col in onehot_encoding_columns:
    unique_values = df[col].unique().tolist()
    result = onehot_encoder\
        .fit_transform(df[col].values.reshape(-1, 1)).toarray()
    df[unique_values] = pd.DataFrame(result, index=df.index)
    df.drop(col, axis=1, inplace=True)
st.write(f'Applied OneHotEncoder for {onehot_encoding_columns}')

st.header('Create simple linear model')
x = df.drop('Survived', axis=1)
y = df['Survived']

with st.spinner("Training model..."):
    with st.echo():
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=.3,
            random_state=42)

        model = LogisticRegression(max_iter=5000)
        model.fit(x_train, y_train)

y_predict = model.predict(x_test)
accuracy = accuracy_score(y_predict, y_test) * 100
st.write('Accuracy of the model is {:3.3f}%'.format(accuracy))

st.header("Model Performance Analysis")

st.subheader("Confusion Matrix")
confussion_matrix = pd.crosstab(y_test, y_predict)
fig = px.imshow(
    confussion_matrix,
    text_auto=True,
    title='Confusion Matrix')
fig.update_layout(
    xaxis_title="Values model predicted",
    yaxis_title="True values",
)
fig

st.subheader("Classification Report with Precision, Recall and F1-Score")
report = classification_report(y_test, y_predict)
st.code('\b ' + report[1:])
