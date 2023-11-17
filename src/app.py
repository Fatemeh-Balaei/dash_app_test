import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from IPython.display import display
from PIL import Image

def fixCoOrdinates( event : pd.Series ) -> pd.Series :
    """
    Apply this method to NHL Data dataframe to rotate cordinate fields by 180
    """
    x, y = event.XCoordinates, event.YCoordinates
    if x > 0:
        x, y = event.XCoordinates * -1.0, event.YCoordinates * -1.0

    x += 50    

    rx = y
    ry = -x

    rx += 42.5
    ry += 50

    return pd.Series( [rx, ry], index=["XCoordinates", "YCoordinates"] )

def setContour(team_bin,contourN = 12):

    #find and set contour layout
    
    maxG = np.max(team_bin)
    minG = np.min(team_bin)

    if abs(maxG) > abs(minG):
        midC = 1/((abs(maxG)/abs(minG))+1)
    else:
        midC = 1-(1/((abs(minG)/abs(maxG))+1))
    
    colorscale = [[0, 'red'],[max(0,midC-0.02), 'white'], [min(1,midC+0.02), 'white'],[1, 'blue']]
    #colorscale = [[0.1, 'rgb(255, 255, 255)'], [0, 'rgb(46, 255, 213)'], [1, 'rgb(255, 28, 251)']]

    return colorscale,minG,maxG

def filter_data(data, season, selected_team):

    filtered_data = data.copy()

    filtered_data = data[data["Season"] == season]
    filtered_data = filtered_data[filtered_data["XCoordinates"].notna()]
    filtered_data = filtered_data[filtered_data["YCoordinates"].notna()]
    filtered_data[['XCoordinates', 'YCoordinates']] = filtered_data[['XCoordinates', 'YCoordinates']].apply(fixCoOrdinates, axis=1, result_type="expand")
    team_list = filtered_data["Team"].unique()
    
    bin = np.zeros((100, 85))
    for index, row in filtered_data.iterrows():
        bin[int(row["YCoordinates"]),int(row["XCoordinates"])] += 1

    team_filtered_data = filtered_data[filtered_data["Team"] == selected_team]
    
    team_bin = np.zeros((100, 85))
    for index, row in team_filtered_data.iterrows():
        team_bin[int(row["YCoordinates"]),int(row["XCoordinates"])] += 1

    team_bin = (team_bin - (bin/len(team_list)))
    team_bin = gaussian_filter(team_bin, sigma=2)

    return team_bin, team_list

path = "json_dataFrame.csv"
data = pd.read_csv(path)

img = Image.open('nhl_rink.png')
nimg = img.crop((0, 0, (img.size[0]/2), img.size[1]))
nimg= nimg.rotate(-90, expand=True)

app = dash.Dash(__name__)
server = app.server

# Define dropdown options
year_options = [20162017, 20172018, 20182019, 20192020, 20202021]
team_options = ['Toronto Maple Leafs']  # Initial team options
contour_options = list(range(1, 21))

# App layout
app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': str(year), 'value': year} for year in year_options],
            value=20162017,
            clearable=False,
            style={'width': '50%'}
        )
    ]),
    html.Div([
        dcc.Dropdown(
            id='team-dropdown',
            options=[{'label': team, 'value': team} for team in team_options],
            value='Toronto Maple Leafs',
            clearable=False,
            style={'width': '50%'}
        ),
        dcc.Slider(
            id='contour-slider',
            min=1,
            max=20,
            step=1,
            value=12,
            marks={i: str(i) for i in range(1, 21)}
        )
    ]),
    dcc.Graph(id='contour-plot')
])

# Callback to update team dropdown options and contour plot
@app.callback(
    [Output('team-dropdown', 'options'),
     Output('team-dropdown', 'value'),
     Output('contour-plot', 'figure')],
    [Input('year-dropdown', 'value'),
     Input('team-dropdown', 'value'),
     Input('contour-slider', 'value')]
)

def update_options_and_plot(selected_year, selected_team, contour_value):
    team_bin, team_list = filter_data(data, selected_year, selected_team)
    colorscale, minG, maxG = setContour(team_bin, contour_value)
    
    # Update team dropdown options and selected value
    team_options = [{'label': team, 'value': team} for team in team_list]
    selected_team = team_list[0] if selected_team not in team_list else selected_team
    
    # Create contour plot
    # fig = go.Figure()
    contour_plot = go.Figure(go.Contour(
        z=team_bin,
        connectgaps=False,
        colorbar=dict(title='Excess shots vs Avg',
                      titleside='right', nticks=contour_value, tickfont_size=10),
        colorscale=colorscale,
        contours=dict(start=minG, end=maxG, size=(maxG-minG)/contour_value),
        line_smoothing=0.85
    ))
    
    # Update contour plot layout
    img = Image.open('nhl_rink.png')
    nimg = img.crop((0, 0, (img.size[0]/2), img.size[1]))
    nimg= nimg.rotate(-90, expand=True)
    contour_plot.update_layout(
        title=f"Shot rates for {selected_team} in {selected_year} vs Avg",
        xaxis_title="Width of the Arena",
        yaxis_title="Length of the Arena from Center",
        images=[dict(
            source=nimg,  # Specify the correct path to your image here
            xref="x",
            yref="y",
            x=0,
            y=100,
            sizex=85,
            sizey=100,
            opacity=0.2,
            layer="above",
            sizing="stretch",
        )]
    )
    return team_options, selected_team, contour_plot

if __name__ == '__main__':
    app.run_server(debug=True)