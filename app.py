import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import json

#Data####################################################################################################################

tournaments = pd.read_csv('tournaments_preparation.csv')
all_match = pd.read_csv('all_match_preparation.csv')


with open('countries.geojson') as f:
    data_geo = json.load(f)


# lists for radar
winner_skills = ['winner_aces', 'winner_break_points_converted', 'winner_break_points_saved', 'winner_double_faults', 'winner_return_points_won', 'winner_total_points_won']
winner_skills_norm = ['winner_aces_norm', 'winner_break_points_converted_norm', 'winner_break_points_saved_norm', 'winner_double_faults_norm', 'winner_return_points_won_norm', 'winner_total_points_won_norm']
loser_skills = ['loser_aces', 'loser_break_points_converted', 'loser_break_points_saved', 'loser_double_faults', 'loser_return_points_won', 'loser_total_points_won']
loser_skills_norm = ['loser_aces_norm', 'loser_break_points_converted_norm', 'loser_break_points_saved_norm', 'loser_double_faults_norm', 'loser_return_points_won_norm', 'loser_total_points_won_norm']
skills = ['Aces', 'Break Points Converted', 'Break Points Saved', 'Double Faults', 'Return Points Won', 'Total Points Won']


#Interactive Components##################################################################################################

slider_year = dcc.Slider(
    id='year_slider',
    min=tournaments['year'].min(),
    max=tournaments['year'].max(),
    # handleLabel={'showCurrentValue': True, 'label': 'year'},
    marks={str(i): '{}'.format(str(i)) for i in
           [2010, 2013, 2016, 2019]},
    value=tournaments['year'].min(),
    step=1
)

region_drop = dcc.Dropdown(
        id = 'drop_region',
        clearable=False, 
        searchable=False, 
        options=[{'label': 'World', 'value': 'World'},
                 {'label': 'Europe', 'value': 'Europe'},
                 {'label': 'Asia', 'value': 'Asia'},
                 {'label': 'Africa', 'value': 'Africa'},
                 {'label': 'North America', 'value': 'North America'},
                 {'label': 'South America', 'value': 'South America'},
                 {'label': 'Oceania', 'value': 'Oceania'}],
        value='World', 
        style= {'margin': '4px', 'box-shadow': '0px 0px #ebb36a', 'border-color': '#ebb36a'}
    )

#------------------------------------------------------ APP ------------------------------------------------------

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([

    html.Div([
        html.H1('Tennis Statistics'),
        html.Img(src=app.get_asset_url('tennis_ball.png'),
                 style={'position': 'relative', 'width': '7%', 'right': '-83px', 'top': '-20px'})
    ], id='1st row for title', className='main_box_style'),

    html.Div([
        html.Div([
    
                html.Div([
                    region_drop,
                    html.Br(),
                        ], id='Region dropdown', style={'height': '10%'}, className='main_box_style'),
    
                html.Div([
                    dcc.Graph(id='bar_plots'),
                        ], id='Bar plots', className='main_box_style'),

                ], id='Bar plot and region dropdown', style={'width': '40%'}, className='main_box_style'),

        html.Div([
            html.Div([
                dcc.Graph(id='scattermap'),
                    ], id='Map', className='main_box_style'),

            html.Div([
                html.Label('Year Slider'),
                slider_year,
                html.Br(),
                    ], id='Slider', style={"height": '10%'}, className='main_box_style'),

                ], id='Map and Slider', style={'width': '60%'}, className='main_box_style')

            ], id='2nd row', style={'display': 'flex'}),

    html.Div([
        html.Div([
            dcc.Graph(id='fig_sunburst'),
        ], id='Sunburst plot', style={'width': '40%'}, className='main_box_style'),

        html.Div([
            html.Div([
    
                # html.Div([
                #     html.H4('Longest Game', style={'font-weight':'normal'}),
                #     html.H3(id='longest_game')
                # ],className='box_stats'),

                # html.Div([
                #     html.H4('Shortest Game', style={'font-weight':'normal'}),
                #     html.H3(id='shortest_game')
                # ],className='box_stats'),
            
                # html.Div([
                #     html.H4('Most Games Won', style={'font-weight':'normal'}),
                #     html.H3(id='most_games_won')
                # ],className='box_stats'),

                # html.Div([
                #     html.H4('Most Frequent Matchup', style={'font-weight':'normal'}),
                #     html.H3(id='most_frequent_atchup')
                # ],className='box_stats'),
            
                # html.Div([
                #     html.H4('Most Break Points Saved', style={'font-weight':'normal'}),
                #     html.H3(id='most_break_points_saved')
                # ],className='box_stats'),

                # html.Div([
                #     html.H4('Most Aces', style={'font-weight':'normal'}),
                #     html.H3(id='most_aces')
                # ],className='box_stats'),
                ], id='Facts', style={'display': 'flex', "height": '10%'}),

            html.Div([
    
                dcc.Graph(id='radar_chart'),
                # if we want a title
                # html.H3('Click on the Map and see the output Bellow (you can use it in the callback [the last one in the app.py]):'),

                ], id='Radar plot', className='main_box_style')
            ], id='Radar and facts', style={'width': '70%'})

        ], id='3rd row', style={'display': 'flex'}),

    html.Div([
        html.H2('Our names:'),
    ], id='4th row for authors', style={'display': 'flex'})
])


#Callbacks####################################################################################################################

@app.callback(
        Output('bar_plots', 'figure'),
        Input('year_slider', 'value'),
        Input('drop_region', 'value')
)

def update_bar_plot(year, region):

    if region != 'World':
        tournaments_filtered = tournaments[(tournaments['year'] == year) & (tournaments['tourney_region'] == region)]
    else:
        tournaments_filtered = tournaments[tournaments['year'] == year]

    # sorting the dataset by prize
    tournaments_filtered = tournaments_filtered.sort_values('tourney_fin_commit_USD', ascending=False)
    # getting the top 10
    tournaments_filtered = tournaments_filtered.iloc[:10,:]

    data_bar_plot = dict(type='bar',
                        x=tournaments_filtered['tourney_name'],
                        y=tournaments_filtered['tourney_fin_commit_USD'])
                        # marker=dict(color=color_dict[tourney_type])

    layout_bar_plot = dict(title=dict(text='Most valuable tournaments',
                                      font=dict(size=24),
                                      x=0.5, y=0.9),
                        yaxis=dict(title='Prize'),
                        font_color='#363535',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)')

    fig_bar = go.Figure(data=data_bar_plot, layout=layout_bar_plot)

    return fig_bar


@app.callback(
    Output('scattermap', 'figure'),
    Output('fig_sunburst', 'figure'),
    Input('year_slider', 'value')
)

def update_plots(year):

    # filter the tournaments dataset by year
    tournaments_filtered = tournaments[tournaments['year'] == year]

    ############---MAP----##############################################################################################
    data_scattermap = dict(type='scattermapbox', 
                           lat=tournaments_filtered['latitude'], 
                           lon=tournaments_filtered['longitude'],
                        
                           mode=['markers', 'lines', 'text'][0],
                           text=tournaments_filtered['tourney_name'], # marker wll show the name of the tournament
                           marker=dict(color='rgb(223, 255, 138)',
                                       opacity = 0.8,
                                       size=10
                                      )
                           )


    layout_scattermap = dict(mapbox=dict(style='white-bg',
                             layers=[dict(source=feature,
                                          below='traces',
                                          type='fill',
                                          fill=dict(outlinecolor='gray')
                                          ) for feature in data_geo['features']]
                            ),
                            title=dict(text='World Map',
                                        x=.5 # Title relative position according to the xaxis, range (0,1)
                                      ),
                            margin=dict(l=0, r=0, b=0, t=30, pad=0),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                            )


    fig_scattermap = go.Figure(data=data_scattermap, layout=layout_scattermap)

    ############---SUBURST----##############################################################################################
    # dataset for the sunburst chart
    tournaments_char = tournaments_filtered.groupby(by=['tourney_type', 'tourney_conditions', 'tourney_surface']).count()[['Unnamed: 0']].rename(columns={'Unnamed: 0': 'Count'})
    tournaments_char['Characteristics'] = 'Characteristics'
    tournaments_char = tournaments_char.reset_index()
    
    fig_sunburst = px.sunburst(tournaments_char, 
                                      path = ['Characteristics', 'tourney_type', 'tourney_conditions', 'tourney_surface'],
                                      values = 'Count',
                                      color_discrete_sequence=px.colors.sequential.Greens_r,
                                      #color_discrete_sequence = ['#F4F5F0','#E6F8F0', '#C0EFE2', '#9CDDCE', '#78CAB9', '#54B6A4', '#2F9F8F', '#1E7B6F','#98ff6e','#80d819','#00ae7d'],


    title = 'Characteristics of the tournament').update_traces(hovertemplate = '%{label}<br>' + 'Number of tournaments: %{value}')

    fig_sunburst = fig_sunburst.update_layout(margin=dict(t=0, l=0, r=0, b=10),
                                                          paper_bgcolor='rgba(0,0,0,0)',
                                                          font_color='#363535'
                                                          )
    return fig_scattermap, fig_sunburst


@app.callback(
    Output('radar_chart', 'figure'),
    Input('scattermap', 'clickData'),
    Input('year_slider', 'value')
)

def update_radar(ClickData, year):

    # extracting the tournament name
    tournament_name = ClickData['points'][0]['text']

    # datasets to plot
    dfwinner_for_plot = pd.DataFrame(all_match[(all_match['tourney_name'] == tournament_name) & (all_match['round_order'] == 1) & (all_match['tourney_year'] == year)][winner_skills_norm].iloc[0])
    dfwinner_for_plot.set_index(pd.Index(skills), inplace = True)
    dfwinner_for_plot.columns = ['score']

    dfwinner_for_text = pd.DataFrame(all_match[(all_match['tourney_name'] == tournament_name) & (all_match['round_order'] == 1) & (all_match['tourney_year'] == year)][winner_skills].iloc[0])
    dfwinner_for_text.set_index(pd.Index(skills), inplace = True)
    dfwinner_for_text.columns = ['score']

    dfloser_for_plot = pd.DataFrame(all_match[(all_match['tourney_name'] == tournament_name) & (all_match['round_order'] == 1) & (all_match['tourney_year'] == year)][loser_skills_norm].iloc[0])
    dfloser_for_plot.set_index(pd.Index(skills), inplace = True)
    dfloser_for_plot.columns = ['score']

    dfloser_for_text = pd.DataFrame(all_match[(all_match['tourney_name'] == tournament_name) & (all_match['round_order'] == 1) & (all_match['tourney_year'] == year)][loser_skills].iloc[0])
    dfloser_for_text.set_index(pd.Index(skills), inplace = True)
    dfloser_for_text.columns = ['score']
    
    list_scores = [dfwinner_for_text.index[i].capitalize() + ': ' + str(dfwinner_for_text['score'][i]) for i in range(len(dfwinner_for_text))]
    text_scores_winner = all_match[(all_match['tourney_name'] == tournament_name) & (all_match['round_order'] == 1) & (all_match['tourney_year'] == year)]['winner_name'].iloc[0]
    for i in list_scores:
        text_scores_winner += '<br>' + i

    list_scores = [dfloser_for_text.index[i].capitalize() + ': ' + str(dfloser_for_text['score'][i]) for i in range(len(dfloser_for_text))]
    text_scores_loser = all_match[(all_match['tourney_name'] == tournament_name) & (all_match['round_order'] == 1) & (all_match['tourney_year'] == year)]['loser_name'].iloc[0]
    for i in list_scores:
        text_scores_loser += '<br>' + i

    ############---RADAR----##############################################################################################
    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r = dfwinner_for_plot['score'],
        theta = dfwinner_for_plot.index,
        fill = 'toself', 
        marker_color = '#00b359',
        opacity = 0.8, 
        hoverinfo = 'text',
        name = text_scores_winner,
        text = [dfwinner_for_text.index[i] + ': ' + str(dfwinner_for_text['score'][i]) for i in range(len(dfwinner_for_text))]
    ))

    fig_radar.add_trace(go.Scatterpolar(
        r = dfloser_for_plot['score'],
        theta = dfloser_for_plot.index,
        fill = 'toself',
        marker_color = '#ff0000',
        hoverinfo = 'text',
        name = text_scores_loser,
        text = [dfloser_for_text.index[i] + ': ' + str(dfloser_for_text['score'][i]) for i in range(len(dfloser_for_text))]
    ))

    fig_radar.update_layout(
        polar = dict(
            bgcolor = 'white',
            radialaxis = dict(
                visible = True,
                type = 'linear',
                autotypenumbers = 'strict',
                autorange = False,
                range = [0, 10],
                angle = 90,
                showline = False,
                showticklabels = False, ticks = '',
                gridcolor = 'black'),
        ),
        width = 730,
        height = 550,
        margin = dict(l = 80, r = 80, t = 20, b = 20),
        showlegend = False,
        template = 'plotly_dark',
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        font_color = 'black',
        font_size = 15
    )

    return fig_radar


if __name__ == '__main__':
    app.run_server(debug=True)