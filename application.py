import dash
from dash import dcc, html
import plotly.graph_objs as go
import pickle
import json
from dash.dependencies import Input, Output, State

########### Define your variables ######
myheading1 = 'Bank Marketing Campaign'
image1 = 'assets/rocauc.html'
tabtitle = 'Bank Marketing'
sourceurl = 'https://archive.ics.uci.edu/ml/datasets/bank+marketing'
githublink = 'https://github.com/ksebastian/logistic-regression'

########### open the json file ######
with open('assets/rocauc.json', 'r') as f:
    fig = json.load(f)

########### open the pickle file ######
filename = open('analysis/bank_marketing_logistic_model.pkl', 'rb')
unpickled_model = pickle.load(filename)
filename.close()

########### list of feature values
job_list = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services',
            'student', 'technician', 'unemployed']
marital_list = ['divorced', 'married', 'single']
education_list = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course',
                  'university.degree']
contact_list = ['cellular', 'telephone']
day_of_week_list = ['fri', 'mon', 'thu', 'tue', 'wed']
### features = 'age', 'job', 'marital', 'education', 'contact', 'day_of_week', 'duration', 'previous'

########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = tabtitle

########### Set up the layout
app.layout = html.Div(children=[
    html.H1(myheading1),

    html.Div([
        html.Div(
            [dcc.Graph(figure=fig, id='fig1')
             ], className='ten columns'),
        html.Div([
            html.H3("Features"),
            html.Div('Age:'),
            dcc.Input(id='age', value=40, type='number', min=18, max=100, step=1),
            html.Div('Job Type:'),
            dcc.Dropdown(
                id='job',
                options=[{'label': i, 'value': i} for i in job_list],
                value=job_list[0]
            ),
            html.Div('Marital Status:'),
            dcc.Dropdown(
                id='marital',
                options=[{'label': i, 'value': i} for i in marital_list],
                value=marital_list[0]
            ),
            html.Div('Education Level:'),
            dcc.Dropdown(
                id='education',
                options=[{'label': i, 'value': i} for i in education_list],
                value=education_list[0]
            ),
            html.Div('Contact Type:'),
            dcc.Dropdown(
                id='contact',
                options=[{'label': i, 'value': i} for i in contact_list],
                value=contact_list[0]
            ),
            html.Div('Contacted Day:'),
            dcc.Dropdown(
                id='day',
                options=[{'label': i, 'value': i} for i in day_of_week_list],
                value=day_of_week_list[0]
            ),
            html.Div('Call Duration:'),
            dcc.Input(id='duration', value=5000, type='number', min=0, max=5000, step=100),
            html.Div('Was called previously(1=Yes, 0=No):'),
            dcc.Input(id='previous', value=0, type='number', min=0, max=1, step=1),
            html.Div('Probability Threshold for Loan Approval'),
            dcc.Input(id='Threshold', value=50, type='number', min=0, max=100, step=1),

        ], className='three columns'),
        html.Div([
            html.H3('Predictions'),
            html.Div('Predicted Status:'),
            html.Div(id='PredResults'),
            html.Br(),
            html.Div('Probability of Approval:'),
            html.Div(id='ApprovalProb'),
            html.Br(),
            html.Div('Probability of Denial:'),
            html.Div(id='DenialProb')
        ], className='three columns')
    ], className='twelve columns',
    ),

    html.Br(),
    html.A('Code on Github', href=githublink),
    html.Br(),
    html.A("Data Source", href=sourceurl),
]
)


######### Define Callback
@app.callback(
    [Output(component_id='PredResults', component_property='children'),
     Output(component_id='ApprovalProb', component_property='children'),
     Output(component_id='DenialProb', component_property='children'),
     ],
    [Input(component_id='age', component_property='value'),
     Input(component_id='job', component_property='value'),
     Input(component_id='marital', component_property='value'),
     Input(component_id='education', component_property='value'),
     Input(component_id='contact', component_property='value'),
     Input(component_id='day', component_property='value'),
     Input(component_id='duration', component_property='value'),
     Input(component_id='previous', component_property='value'),
     Input(component_id='Threshold', component_property='value')
     ])
def prediction_function(age, job, marital, education, contact, day, duration, previous, Threshold):
    try:
        data = [[age, job, marital, education, contact, day, duration, previous]]
        print("Data:{}", data)
        rawprob = 100 * unpickled_model.predict_proba(data)[0][1]
        print("Raw Probability:{}", rawprob)

        func = lambda y: 'Approved' if int(rawprob) > Threshold else 'Denied'
        formatted_y = func(rawprob)
        print("Predicted Status:{}", data)

        deny_prob = unpickled_model.predict_proba(data)[0][0] * 100
        formatted_deny_prob = "{:,.2f}%".format(deny_prob)
        print("Denial Probability:{}", formatted_deny_prob)

        app_prob = unpickled_model.predict_proba(data)[0][1] * 100
        formatted_app_prob = "{:,.2f}%".format(app_prob)
        print("Approval Probability:{}", formatted_app_prob)

        return formatted_y, formatted_app_prob, formatted_deny_prob
    except:
        return "inadequate inputs", "inadequate inputs", "inadequate inputs"


############ Deploy
if __name__ == '__main__':
    app.run_server(debug=True)
