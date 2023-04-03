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
        options=[{'label': 'World', 'value': 'world'},
                 {'label': 'Europe', 'value': 'europe'},
                 {'label': 'Asia', 'value': 'asia'},
                 {'label': 'Africa', 'value': 'africa'},
                 {'label': 'North America', 'value': 'north america'},
                 {'label': 'South America', 'value': 'south america'},
                 {'label': 'Oceania', 'value': 'oceania'}],
        value='world', 
        style= {'margin': '4px', 'box-shadow': '0px 0px #ebb36a', 'border-color': '#ebb36a'}
    )

#App########################################################################################################################

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
    
                html.Div([
                    html.H4('Longest Game (in minutes)', style={'font-weight':'normal'}),
                    html.H3(id='longest_game')
                ],className='box_stats'),

                html.Div([
                    html.H4('Shortest Game (in minutes)', style={'font-weight':'normal'}),
                    html.H3(id='shortest_game')
                ],className='box_stats'),
            
                html.Div([
                    html.H4('Most Games Won', style={'font-weight':'normal'}),
                    html.H3(id='most_games_won')
                ],className='box_stats'),

                html.Div([
                    html.H4('Most Frequent Matchup', style={'font-weight':'normal'}),
                    html.H3(id='most_frequent_matchup')
                ],className='box_stats'),
            
                html.Div([
                    html.H4('Most Break Points Saved', style={'font-weight':'normal'}),
                    html.H3(id='most_break_points_saved')
                ],className='box_stats'),

                html.Div([
                    html.H4('Most Aces', style={'font-weight':'normal'}),
                    html.H3(id='most_aces')
                ],className='box_stats'),

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
        Output('scattermap', 'figure'),
        Output('fig_sunburst', 'figure'),
        Input('year_slider', 'value'),
        Input('drop_region', 'value')
)

def update_bar_plot(year, region):

    if region != 'world':
        tournaments_filtered = tournaments[(tournaments['year'] == year) & (tournaments['tourney_region'] == region)]
    else:
        tournaments_filtered = tournaments[tournaments['year'] == year]

    ############---BAR PLOT----##############################################################################################

    # sorting the dataset by prize
    tournaments_filtered_top = tournaments_filtered.sort_values('tourney_fin_commit_USD', ascending=False)
    
    # getting the top 10
    tournaments_filtered_top = tournaments_filtered.iloc[:10,:]
    
    data_bar_plot = dict(type='bar',
                        x=tournaments_filtered_top['tourney_name'],
                        y=tournaments_filtered_top['tourney_fin_commit_USD'],
                        marker_color=['#036666' if x=='Grand Slam' else '#248277' if x=='Masters 1000' else '#358f80' if x=='ATP Finals' else '#56ab91' if x=='ATP 500' else '#78c6a3' if x=='ATP 250' else '#99e2b4' if x=='Next Gen Finals' else '#000000' for x in tournaments_filtered['tourney_type']])

    layout_bar_plot = dict(title=dict(text='Most valuable tournaments',
                                      font=dict(size=24),
                                      x=0.5, y=0.9),
                        yaxis=dict(title='Prize'),
                        font_color='#363535',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)')

    fig_bar = go.Figure(data=data_bar_plot, layout=layout_bar_plot)

    ############---MAP----##############################################################################################
    data_scattermap = dict(type='scattergeo', 
                           lat=tournaments_filtered['latitude'], 
                           lon=tournaments_filtered['longitude'],
                           mode='markers',
                           text=tournaments_filtered['tourney_name'], #marker wll show the name of the tournament
                           marker=dict(color=['#036666' if x=='Grand Slam' else '#248277' if x=='Masters 1000' else '#358f80' if x=='ATP Finals' else '#56ab91' if x=='ATP 500' else '#78c6a3' if x=='ATP 250' else '#99e2b4' if x=='Next Gen Finals' else '#000000' for x in tournaments_filtered['tourney_type']],
                                       opacity = 1,
                                       size=10
                                      )
                           )

    layout_scattermap = dict(geo=dict(scope=region,
                                     showcountries=True,
                                     projection=dict(type='natural earth'),
                                     bgcolor= 'rgba(0,0,0,0)',
                                     landcolor='#cccccc'),
                            title=dict(text='Map',
                                        x=.5 # Title relative position according to the xaxis, range (0,1)
                                      ),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                            )

    fig_scattermap = go.Figure(data=data_scattermap, layout=layout_scattermap)

    ############---SUBURST----##############################################################################################
    # dataset for the sunburst chart
    tournaments_char = tournaments_filtered.groupby(by=['tourney_type', 'tourney_conditions', 'tourney_surface']).count()[['Unnamed: 0']].rename(columns={'Unnamed: 0': 'Count'})
    tournaments_char['Characteristics'] = 'Characteristics'
    tournaments_char = tournaments_char.reset_index()

    colour_type={'Grand Slam': '#036666',
                'Masters 1000': '#248277',
                'ATP Finals': '#358f80',
                'ATP 500': '#56ab91',
                'ATP 250': '#78c6a3',
                'Next Gen Finals': '#99e2b4',
                'Characteristics': 'rgba(0,0,0,0)'}

    fig_sunburst = px.sunburst(tournaments_char, 
                                path = ['Characteristics', 'tourney_type', 'tourney_conditions', 'tourney_surface'],
                                values = 'Count',
                                color=tournaments_char['tourney_type'],
                                # category_orders={'tourney_type': ['Grand Slam', 'Masters 1000', 'ATP Finals', 'ATP 500', 'ATP 250', 'Next Gen Finals']},
                                # color_discrete_sequence=['#036666', '#248277', '#358f80', '#56ab91', '#78c6a3','#99e2b4', 'rgba(0,0,0,0)'],
                                color_discrete_map=colour_type,
                                #color_discrete_sequence = ['#F4F5F0','#E6F8F0', '#C0EFE2', '#9CDDCE', '#', '#54B6A4', '#2F9F8F', '#1E7B6F','#98ff6e','#80d819','#00ae7d'],
                                title = 'Characteristics of the tournaments').update_traces(hovertemplate = '%{label}<br>' + 'Number of tournaments: %{value}', branchvalues='total')

    fig_sunburst = fig_sunburst.update_layout(margin=dict(t=0, l=0, r=0, b=10),
                                                          paper_bgcolor='rgba(0,0,0,0)',
                                                          font_color='#363535'
                                                          )
    
    return fig_bar, fig_scattermap, fig_sunburst


# @app.callback(
#     #Output('scattermap', 'figure'),
#     Output('fig_sunburst', 'figure'),
#     Input('year_slider', 'value')
# )

# def update_plots(year):

    
#     # return fig_scattermap, fig_sunburst


@app.callback(
    Output('longest_game', 'children'),
    Output('shortest_game', 'children'),
    Output('most_games_won', 'children'),
    Output('most_frequent_matchup', 'children'),
    Output('most_break_points_saved', 'children'),
    Output('most_aces', 'children'),
    Output('radar_chart', 'figure'),

    Input('scattermap', 'clickData'),
    Input('year_slider', 'value')
)

def update_stats_radar(ClickData, year):

    # defining the default values
    if ClickData:
        # extracting the tournament name
        tournament_name=ClickData['points'][0]['text']
    else:
        tournament_name='Wimbledon'
        year=2019

    # filtering the datasets for the tournament that is going to be displayed
    all_match_tournament=all_match[all_match['tourney_name'] == tournament_name]
    tournaments_tournament=tournaments[tournaments['tourney_name'] == tournament_name]

    # statistics
    longest_game_str=str(round(all_match_tournament['match_duration'].max(),0))[:-2]
    shortest_game_str=str(round(all_match_tournament['match_duration'].min(),0))[:-2]
    most_games_won_str=str(tournaments_tournament['singles_winner_name'].value_counts()[0])
    most_frequent_matchup_str=str(all_match_tournament['pair'].value_counts().index[0])
    most_break_points_saved_str=str(round(all_match_tournament['max_break_points_saved'].max(),0))[:-2]
    most_aces_str=str(round(all_match_tournament['max_aces'].max(),0))[:-2]


    # filtering the dataset for the game that is going to be plotted
    all_match_final=all_match[(all_match['tourney_name'] == tournament_name) & (all_match['round_order'] == 1) & (all_match['tourney_year'] == year)]

    # datasets to plot
    winner_for_plot = pd.DataFrame(all_match_final[winner_skills_norm].iloc[0])
    winner_for_plot.set_index(pd.Index(skills), inplace = True)
    winner_for_plot.columns = ['score']

    winner_for_text = pd.DataFrame(all_match_final[winner_skills].iloc[0])
    winner_for_text.set_index(pd.Index(skills), inplace = True)
    winner_for_text.columns = ['score']

    loser_for_plot = pd.DataFrame(all_match_final[loser_skills_norm].iloc[0])
    loser_for_plot.set_index(pd.Index(skills), inplace = True)
    loser_for_plot.columns = ['score']

    loser_for_text = pd.DataFrame(all_match_final[loser_skills].iloc[0])
    loser_for_text.set_index(pd.Index(skills), inplace = True)
    loser_for_text.columns = ['score']
    
    list_scores = [winner_for_text.index[i].capitalize() + ': ' + str(winner_for_text['score'][i]) for i in range(len(winner_for_text))]
    text_scores_winner = all_match_final['winner_name'].iloc[0]
    for i in list_scores:
        text_scores_winner += '<br>' + i

    list_scores = [loser_for_text.index[i].capitalize() + ': ' + str(loser_for_text['score'][i]) for i in range(len(loser_for_text))]
    text_scores_loser = all_match_final['loser_name'].iloc[0]
    for i in list_scores:
        text_scores_loser += '<br>' + i

    ############---RADAR----##############################################################################################
    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(r = winner_for_plot['score'],
                                        theta = winner_for_plot.index,
                                        fill = 'toself', 
                                        marker_color = '#00b359',
                                        opacity = 0.8, 
                                        hoverinfo = 'text',
                                        name = text_scores_winner,
                                        text = [winner_for_text.index[i] + ': ' + str(winner_for_text['score'][i]) for i in range(len(winner_for_text))]
                        ))

    fig_radar.add_trace(go.Scatterpolar(r = loser_for_plot['score'],
                                        theta = loser_for_plot.index,
                                        fill = 'toself',
                                        marker_color = '#ff0000',
                                        hoverinfo = 'text',
                                        name = text_scores_loser,
                                        text = [loser_for_text.index[i] + ': ' + str(loser_for_text['score'][i]) for i in range(len(loser_for_text))]
                        ))

    fig_radar.update_layout(polar = dict(bgcolor = 'rgba(0, 0, 0, 0)',
                                         radialaxis = dict(visible = True,
                                                           type = 'linear',
                                                           autotypenumbers = 'strict',
                                                           autorange = True,
                                                           # range = [0, 10],
                                                           angle = 90,
                                                           showline = False,
                                                           showticklabels = False, ticks = '',
                                                           gridcolor = 'black'
                                                           ),
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

    return longest_game_str, shortest_game_str, most_games_won_str, most_frequent_matchup_str, most_break_points_saved_str, most_aces_str, fig_radar


if __name__ == '__main__':
    app.run_server(debug=True)