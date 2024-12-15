import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Sample data
df = pd.DataFrame({
    "Category": ["A", "B", "C", "D"] * 10,
    "Value": [i + (j % 4) * 2 for i in range(10) for j in range(4)],
    "Index": list(range(10)) * 4
})

# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Simple Dash App"),
    html.Label("Select Category:"),
    dcc.Dropdown(
        id="category-dropdown",
        options=[{"label": cat, "value": cat} for cat in df["Category"].unique()],
        value="A"
    ),
    dcc.Graph(id="line-chart")
])

# Callbacks
@app.callback(
    Output("line-chart", "figure"),
    Input("category-dropdown", "value")
)
def update_chart(selected_category):
    filtered_df = df[df["Category"] == selected_category]
    fig = px.line(filtered_df, x="Index", y="Value", title=f"Category {selected_category} Line Chart")
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
