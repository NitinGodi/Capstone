import base64
import io
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import feedback_tool

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('capstone', external_stylesheets=external_stylesheets)

my_app.layout = html.Div([
    html.H1('ESSAY STRUCTURE IDENTIFIER', style={'textAlign': 'center'}),
    html.Br(),
    html.Div([
        html.Div([html.H3('INPUT', style={'align':'center'}),
                  html.Br(),
                  html.H5('Select Model:'),
                  dcc.Dropdown(id='model-selection',
                               options=[
                                   {'label':'Naive Bayes', 'value':'naive-bayes'},
                                   {'label':'LSTM', 'value':'lstm'},
                                   {'label':'Roberta', 'value':'roberta'},
                               ],
                               style={'width':'80%'}),
                  html.Br(),
                  html.H5('Upload text file:'),
                  html.Br(),
                  dcc.Upload(html.Button('Upload file',id='upload-button'),id='file-input'),
                  html.Br(),
                  html.H5('OR Type essay here:'),
                  dcc.Textarea(id='input-text',style={'width':'85%','height':600}),
                  html.Br(),
                  html.Button('Submit',id='manual-input',n_clicks=0)
                  ], style={'width':'30%','float':'left'}),
        html.Div([html.H3('OUTPUT', style={'align':'center'}),
                  html.Br(),
                  html.P(id='output', style={'text-align':'justify'})
                  ], style={'width':'50%','float':'left'}),
        html.Div([html.H4('Legend:', style={'color':'Black','text-align':'right'}),
                  html.H5('No Class', style={'color':'Black','text-align':'right'}),
                  html.H5('Lead', style={'color':'green','text-align':'right'}),
                  html.H5('Position', style={'color':'red','text-align':'right'}),
                  html.H5('Claim', style={'color':'blue','text-align':'right'}),
                  html.H5('Counterclaim', style={'color':'orange','text-align':'right'}),
                  html.H5('Rebuttal', style={'color':'yellow','text-align':'right'}),
                  html.H5('Evidence', style={'color':'purple','text-align':'right'}),
                  html.H5('Concluding Statement', style={'color':'magenta','text-align':'right'})
                  ], style={'width':'10%','float':'right','margin-left':'5%'})
    ])
],style={'width':'90%','margin-left':'5%', 'margin-right':'5%'})

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    data = io.StringIO(decoded.decode('utf-8')).readlines()
    return data

cls_to_color_mapper = {'No Class':'black', 'Lead':'green', 'Position':'red', 'Claim':'blue', 'Counterclaim':'orange', 'Rebuttal':'yellow', 'Evidence':'purple', 'Concluding Statement':'magenta'}

def highlight(essay, model):
    result = feedback_tool.get_result(model, essay)
    split_essay = essay.split()
    cls, predstring = result['class'].values, result['predictionstring'].values
    display_text=[]
    if predstring[0][0] != 0:
        display_text.append(html.Span(' '.join(split_essay[:predstring[0][0]]) + ' ', style={'color':cls_to_color_mapper['No Class']}))
    last = predstring[0][0]
    for idx in range(len(predstring)):
        if last == predstring[idx][0]:
            display_text.append(html.Span(' '.join(split_essay[predstring[idx][0]:predstring[idx][-1]+1]) + ' ', style={'color':cls_to_color_mapper[cls[idx]]}))
            last = predstring[idx][-1]+1
        else:
            display_text.append(html.Span(' '.join(split_essay[last:predstring[idx][0]]) + ' ', style={'color': cls_to_color_mapper['No Class']}))
            display_text.append(html.Span(' '.join(split_essay[predstring[idx][0]:predstring[idx][-1]+1]) + ' ', style={'color': cls_to_color_mapper[cls[idx]]}))
            last = predstring[idx][-1] + 1
    if last < len(split_essay):
        display_text.append(html.Span(' '.join(split_essay[last:]), style={'color': cls_to_color_mapper['No Class']}))
    return display_text

@my_app.callback(Output(component_id='output', component_property='children'),
                 Input(component_id='model-selection', component_property='value'),
                 Input(component_id='file-input', component_property='filename'),
                 Input(component_id='file-input', component_property='contents'),
                 Input(component_id='manual-input', component_property='n_clicks'),
                 State(component_id='input-text', component_property='value')
                 )
def update_display1(model,filename, contents,n,txt):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'file-input':
        x = parse_contents(contents)
        disp = ''.join(x)
        return highlight(disp, model)
    elif button_id == 'manual-input':
        return highlight(txt, model)

my_app.run_server(
    port=8052,
    host='0.0.0.0'
)

