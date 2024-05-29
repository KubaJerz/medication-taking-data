import dash
import json
import pandas as pd
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import plotly
import sys
import os

app = dash.Dash(__name__)

themes = ['plotly_dark', 'seaborn', 'plotly']
vline_width = 2

if len(sys.argv) == 2:
    data_filename = sys.argv[1]
    theme = themes[0]
    backgroundcolor = '#121212'
    port = 8050  # Default port
elif len(sys.argv) == 3:
    data_filename = sys.argv[1]
    try:
        port = int(sys.argv[2])
    except ValueError:
        theme = themes[int(sys.argv[2])]
        if sys.argv[2] != '0':
            backgroundcolor = 'white'
        else:
            backgroundcolor = '#121212'
        port = 8050  # Default port
    else:
        theme = themes[0]
        backgroundcolor = '#121212'
elif len(sys.argv) == 4:
    data_filename = sys.argv[1]
    try:
        port = int(sys.argv[2])
    except ValueError:
        port = 8050  # Default port
    theme = themes[int(sys.argv[3])]
    if sys.argv[3] != '0':
        backgroundcolor = 'white'
    else:
        backgroundcolor = '#121212'
else:
    data_filename = 'acceleration.csv'
    theme = themes[0]
    backgroundcolor = '#121212'
    port = 8050  # Default port

# Load CSV data
acc_data = pd.read_csv(data_filename, skiprows=1)
acc_data['timestamp'] = ((acc_data['timestamp'] - acc_data['timestamp'].iloc[0]) / 1e9)

#split up data is its too big
#data_length = len(acc_data)
#start_index = int(data_length * 15  / 16)  # Index to start plotting from
acc_data = acc_data.iloc[::5, :]

# Initial button color and option
button_color = 'blue'
options = ['left_water', 'left_listerine', 'right_water', 'right_listerine']
current_option_index = 0
current_option = options[current_option_index]
start_time = None
end_time = None

# init dict
data = {
    "left": {
        "water": [],
        "listerine": []
    },
    "right": {
        "water": [],
        "listerine": []
    }
}

# init the stack
stack = []

app.layout = html.Div([
    html.Button(id='btn', children=current_option, style={'backgroundColor': button_color, 'color': 'white'}), 
    html.Button(id='save_btn', children='save btn', style={'position': 'absolute', 'top': '8px', 'right': '50px'}),
    html.Button(id='undo_btn', children='undo', style={'position': 'absolute', 'top': '8px', 'right': '550px'}),
    dcc.Graph(
        id='graph',
        figure=px.line(acc_data, x='timestamp', y='x', title=data_filename).update_layout(
            xaxis_title='Time (s)',
            yaxis_title='Acceleration (m/s²)',
            width=1800,
            height=900
        ).add_trace(go.Scatter(x=acc_data['timestamp'], y=acc_data['y'], mode='lines', name='y'))
         .add_trace(go.Scatter(x=acc_data['timestamp'], y=acc_data['z'], mode='lines', name='z'))
         .update_layout(template=theme)
    ),
    dcc.Store(id='current_option', data=current_option),
    dcc.Store(id='current_option_index', data=current_option_index),
    dcc.Store(id='start_time', data=None),
    dcc.Store(id='end_time', data=None),
    dcc.ConfirmDialog(id='err_msg', message="You have to select start and end time before you can switch action and hand. You have not selected end time."),
], style={'backgroundColor': backgroundcolor, 'width': '100vw', 'height': '100vh', 'body': { 'margin': '0', 'padding': '0', 'overflow': 'hidden'}})

@app.callback(
    Output('graph', 'figure', allow_duplicate=True),
    Output('start_time', 'data'),
    Output('end_time', 'data'),
    Input('graph', 'clickData'),
    State('current_option', 'data'),
    State('start_time', 'data'),
    State('end_time', 'data'),
    State('graph', 'figure'),
    prevent_initial_call=True
)
def process_clk(click_data, current_option, start_time, end_time, figure):
    if click_data:
        x_value = click_data['points'][0]['x']
        if start_time is None:
            option_parts = current_option.split('_')
            side = option_parts[0]
            action = option_parts[1]

            start_time = x_value
            stack.append([side, action])
        else:
            end_time = x_value
            option_parts = current_option.split('_')
            side = option_parts[0]
            action = option_parts[1]
            stack.append([side, action])
            data[side][action].append({'start': start_time, 'end': end_time})
            start_time = None
            end_time = None

        # This code adds the vertical line
        fig = go.Figure(figure)  # because figure is a dict we need it as a plotly object
        fig.add_vline(x=x_value, line_width=vline_width, line_dash="dash", line_color=button_color)
        figure = fig.to_dict()  # convert the updated fig back to a dictionary

    return figure, start_time, end_time

#removes the v lines when pressing undo
def remove_vline(fig, x_value):
    shapes = fig.layout.shapes or []
    updated_shapes = [shape for shape in shapes if shape.x0 != x_value]
    if updated_shapes:
        fig.update_layout(shapes=updated_shapes)
    else:
        #super janky way to do this needs to be better but if you try to update the fig with empty array for shapes it does not work
        fig.add_vline(x=x_value, line_width=vline_width+1, line_color=backgroundcolor)
    return fig

#this removes the data from the dic when using undo and then calls the remove v line function
@app.callback(
    Output('start_time', 'data', allow_duplicate=True),
    Output('graph', 'figure', allow_duplicate=True),
    Input('undo_btn', 'n_clicks'),
    State('start_time', 'data'),
    State('graph', 'figure'),
    prevent_initial_call=True
)
def undo_func(n_clicks, start_time, figure):
    if n_clicks:
        if start_time is not None:
            if stack:
                fig = go.Figure(figure)
                fig = remove_vline(fig, start_time)
                start_time = None
                figure = fig.to_dict()
            return start_time, figure
        else:
            if stack:
                element0 = stack.pop()
                element1 = stack.pop()
                vline0_corr = data[element0[0]][element0[1]][-1]['end']
                vline1_corr = data[element1[0]][element1[1]][-1]['start']
                fig = go.Figure(figure)
                fig = remove_vline(fig, vline0_corr)
                fig = remove_vline(fig, vline1_corr)
                del data[element0[0]][element0[1]][-1]
                figure = fig.to_dict()
            return start_time, figure

@app.callback(
    Output('btn', 'children'),
    Output('btn', 'style'),
    Output('current_option', 'data'),
    Output('current_option_index', 'data'),
    Output('err_msg', 'displayed'),
    Input('btn', 'n_clicks'),
    State('current_option_index', 'data'),
    State('start_time', 'data')
)
def change_button_color(n_clicks, current_option_index, start_time):
    global button_color, current_option
    
    if n_clicks:
        if start_time is not None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, True
        
        button_color = 'green' if button_color == 'blue' else 'blue'
        current_option_index = (current_option_index + 1) % len(options)
        current_option = options[current_option_index]
        
    return current_option, {'backgroundColor': button_color, 'color': 'white'}, current_option, current_option_index, False

def write_data_to_file():
    if len(sys.argv) > 1:
        dir_path = os.path.dirname(sys.argv[1])
        new_file_path = os.path.join(dir_path, 'labels.json')
        with open(new_file_path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        with open('labels.json', 'w') as f:
            json.dump(data, f, indent=2)

@app.callback(
    Input('save_btn', 'n_clicks'),
    prevent_initial_call=True
)
def exit_app(n_clicks):
    if n_clicks:
        write_data_to_file()

if __name__ == '__main__':
    app.run_server(debug=True, port=port)
