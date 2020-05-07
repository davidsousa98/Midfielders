# Import libraries
import pandas as pd
from datetime import date, datetime
import plotly.offline as pyo
from plotly import graph_objs as go
from plotly.subplots import make_subplots

# Import dataset
df = pd.read_excel('/Users/davidsousa/Documents/SportsDS/datasets/midfielders.xlsx')

# Retrieve age from date of birth
def calculate_age(born):
    born = datetime.strptime(born, "%Y-%m-%d").date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

df['Age'] = df['BirthDate'].apply(calculate_age)
df.drop(columns='BirthDate', inplace=True)


# Pass success rate visualization
df['incompletepasses_per_game'] = df['passes_per_game'] - df['completepasses_per_game']

top_successpasses = df.loc[df['total_matches'] > 19]
top_successpasses = top_successpasses.sort_values(by='passes_per_game', ascending=False).head(10)
top_successpasses['incompletepasses_per_game'] = top_successpasses['incompletepasses_per_game'].round()
top_successpasses['completepasses_per_game'] = top_successpasses['completepasses_per_game'].round()
top_successpasses['%passucessrate_per_game'] = top_successpasses['%passucessrate_per_game'].round(2)

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(
        x=top_successpasses['PlayerName'], y=top_successpasses["%passucessrate_per_game"], name="pass success rate",
        marker=dict(color='rgba(0, 0, 0, 1)'), line=dict(dash='dot')),
        secondary_y=True
    )

fig.add_trace(
    go.Bar(
        x=top_successpasses['PlayerName'], y=top_successpasses["completepasses_per_game"], name="complete passes",
        marker=dict(color='rgba(240, 52, 52, 1)'), text=top_successpasses["completepasses_per_game"], textposition='inside'),
        secondary_y=False
    )

fig.add_trace(
    go.Bar(
        x=top_successpasses['PlayerName'], y=top_successpasses["incompletepasses_per_game"], name="incomplete passes",
        marker=dict(color='rgba(226, 106, 106, 1)'), text=top_successpasses["incompletepasses_per_game"], textposition='inside'),
        secondary_y=False
    )

fig.update_yaxes(title_text="Average number of passes per game", secondary_y=False)
fig.update_yaxes(title_text="Average pass success rate per game (%)", secondary_y=True)
fig.update_layout(title='Pass success rate analysis', xaxis_title="Player", barmode='stack')

pyo.plot(fig)



smartpasses_df = df.sort_values(by='smartpasses_per_game', ascending=False).head(10)

fig = go.Figure(
    go.Bar(
        y=smartpasses_df['PlayerName'], x=smartpasses_df["smartpasses_per_game"], name="smart passes",
        marker=dict(color='rgba(240, 52, 52, 1)'), text=smartpasses_df["smartpasses_per_game"], textposition='inside', orientation='h')
    )
pyo.plot(fig)