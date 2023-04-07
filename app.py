import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

#Data####################################################################################################################

tournaments = pd.read_csv('tournaments_preparation.csv')
all_match = pd.read_csv('all_match_preparation.csv')

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
    marks={str(i): '{}'.format(str(i)) for i in
           [2010, 2013, 2016, 2019]},
    value=tournaments['year'].min(),
    step=1
)

region_drop = dcc.Dropdown(
        id='drop_region',
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
        style= {'margin': '4px', 'box-shadow': '0px 0px #B7EAFF', 'border-color': '#B7EAFF'}
    )

#App########################################################################################################################

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([

    html.Div([
        html.H1('Tennis Statistics'),
        html.Br(),
        html.Label('With the aim of getting a better understanding of Tennis and its tournaments around the world, this interactive dashboard provides insights into key performance indicators, characteristics of each type of tournament and player statistics.'),
    ], id='1st row for title', className='main_box_style'),

    html.Div([
        html.Div([
            html.Div([
                html.H2('Tournaments by Prize (US dollars)'),    
                dcc.Graph(id='bar_plots'),
                ], id='Bar plot', className='main_box_style'),

                html.Div([
                    html.Img(src=app.get_asset_url('boys-playing.png'),style={'position': 'relative'})
                ], id='Boys image'),

                html.Div([
                    html.H2('Tournaments by Characteristics'),
                    html.Br(),
                    html.Label("Slide the cursor over the chart for the number of tournaments."),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    dcc.Graph(id='fig_sunburst'),
                ], id='Sunburst', className='main_box_style'),

            ], id='Left Body', style={'width': '40%'}),

        html.Div([
                html.Div([
                    html.Br(),
                    html.Label('Select one region and one year to watch the dashboard change.'),
                    html.Br(),
                    html.Br(),
                    html.Br(),

                    html.Div([
                        html.Div([
                            region_drop,
                            html.Br(),
                        ], id='Region dropdown', style={'height': '10%', 'width': '35%'}),

                        html.Div([
                            html.Label('Year Slider'),
                            slider_year,
                            html.Br(),
                        ], id='Slider', style={'height': '10%', 'width': '65%'})

                    ], id='Slider and Dropdown', style={'display': 'flex'})
                ], id='Slider and Dropdown and title', className='main_box_style'),

                html.Div([
                    html.H2('Tournaments Map'),
                    html.Br(),
                    html.Label("Select one tournament to see some facts and the final's match statistics."),
                    dcc.Graph(id='scattermap'),
                ], id='Map', className='main_box_style'),

                html.Div([
                    
                    html.Div([
                        html.H2('Tournament Facts')
                    ], id='Title', style={'text-align': 'center'}),

                    html.Div([
                        html.Div([
                            html.H3('Longest Game (in minutes)'),
                            html.H4(id='longest_game')
                        ],className='box_stats'),

                        html.Div([
                            html.H3('Shortest Game (in minutes)'),
                            html.H4(id='shortest_game')
                        ],className='box_stats'),
                    
                        html.Div([
                            html.H3('Most Finals Won'),
                            html.H4(id='most_tournaments_won')
                        ],className='box_stats'),

                        html.Div([
                            html.H3('Most Frequent Matchup'),
                            html.H4(id='most_frequent_matchup')
                        ],className='box_stats'),
                    
                        html.Div([
                            html.H3('Most Break Points Saved'),
                            html.H4(id='most_break_points_saved')
                        ],className='box_stats'),

                        html.Div([
                            html.H3('Most Aces'),
                            html.H4(id='most_aces')
                        ],className='box_stats'),

                    ], id='Facts', style={'display': 'flex', 'margin': 'auto'})

                ], id='Facts and title'),

                html.Br(),

                html.Div([
                    html.H2('Tournament Final Statistics'),
                    html.Br(),
                    html.Label(id='radar_legend'),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    dcc.Graph(id='radar_chart'),
                ], id='Radar', className='main_box_style')

            ], id='Right Body', style={'width': '60%'}),
    ], id='App body', style={'display': 'flex'}),

    html.Div([
        html.Div([
            html.H3('Project made by:', style={'font-size': 'medium', 'text-align': 'left'}),
            html.Label('Ana Mendonça (20220678), Beatriz Sousa (20220674), Cláudia Rocha (R20191249), Susana Dias (20220198)'),
        ], id='Authors', style={'width': '70%'}),

        html.Div([
            html.H3('Sources:', style={'font-size': 'medium', 'text-align': 'left'}),
            html.A('GitHub', href='https://github.com/serve-and-volley/atp-world-tour-tennis-data', target='_blank'),
        ], id='Sources', style={'width': '30%'})

    ], id='Authors and Sources', style={'display': 'flex'})
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
    tournaments_filtered_top = tournaments_filtered_top.iloc[:10,:]
    
    data_bar_plot = dict(type='bar',
                        x=tournaments_filtered_top['tourney_name'],
                        y=tournaments_filtered_top['tourney_fin_commit_USD'],
                        text=tournaments_filtered_top['tourney_fin_commit_USD'],
                        hoverinfo='text',
                        textposition = 'none',
                        marker_color=['#75c2a2' if x=='Grand Slam' else '#b7eaff' if x=='Masters 1000' else '#548570' if x=='ATP Finals' else '#76B4CD' if x=='ATP 500' else '#37648E' if x=='ATP 250' else '#a0bb9a' if x=='Next Gen Finals' else '#000000' for x in tournaments_filtered_top['tourney_type']])

    layout_bar_plot = dict(margin=dict(t=0, l=0, r=0, b=10),
                           yaxis=dict(title='Prize'),
                           font_color='#363535',
                           paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)',
                           font_family='Bahnschrift Light',
                           hoverlabel=dict(font_family='Bahnschrift Light'))

    fig_bar = go.Figure(data=data_bar_plot, layout=layout_bar_plot)

    ############---MAP----##############################################################################################
    data_scattermap = dict(type='scattergeo', 
                           lat=tournaments_filtered['latitude'], 
                           lon=tournaments_filtered['longitude'],
                           mode='markers',
                           text=tournaments_filtered['tourney_name'], #marker wll show the name of the tournament
                           hoverinfo='text', #only show the tournament name on hover
                           marker=dict(color=['#75c2a2' if x=='Grand Slam' else '#b7eaff' if x=='Masters 1000' else '#548570' if x=='ATP Finals' else '#76B4CD' if x=='ATP 500' else '#37648E' if x=='ATP 250' else '#a0bb9a' if x=='Next Gen Finals' else '#000000' for x in tournaments_filtered['tourney_type']],
                                       opacity = 1,
                                       size=10
                                      ),
                           )

    layout_scattermap = dict(geo=dict(scope=region,
                                     showcountries=True,
                                     projection=dict(type='natural earth'),
                                     bgcolor= 'rgba(0,0,0,0)',
                                     landcolor='#cccccc'),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            hoverlabel=dict(font_family='Bahnschrift Light'),
                            margin=dict(t=0, l=4, r=4, b=10)
                            )

    fig_scattermap = go.Figure(data=data_scattermap, layout=layout_scattermap)

    ############---SUBURST----##############################################################################################
    # dataset for the sunburst chart
    tournaments_char = tournaments_filtered.groupby(by=['tourney_type', 'tourney_conditions', 'tourney_surface']).count()[['Unnamed: 0']].rename(columns={'Unnamed: 0': 'Count'})
    tournaments_char['Characteristics'] = 'Characteristics'
    tournaments_char = tournaments_char.reset_index()

    colour_type={'Grand Slam': '#75c2a2',
                'Masters 1000': '#b7eaff',
                'ATP Finals': '#548570',
                'ATP 500': '#76B4CD',
                'ATP 250': '#37648E',
                'Next Gen Finals': '#a0bb9a',
                '?': 'rgba(0,0,0,0)'}

    fig_sunburst = px.sunburst(tournaments_char, 
                                path = ['tourney_type', 'tourney_conditions', 'tourney_surface'],
                                values = 'Count',
                                color=tournaments_char['tourney_type'],
                                color_discrete_map=colour_type,
                                title = 'Characteristics of the tournaments').update_traces(hovertemplate = '%{label}<br>' + 'Number of tournaments: %{value}', branchvalues='total', marker=dict(line=dict(width=1.5, color='white')))

    fig_sunburst = fig_sunburst.update_layout(margin=dict(t=0, l=0, r=0, b=10),
                                                          paper_bgcolor='rgba(0,0,0,0)',
                                                          font_color='#363535',
                                                          font_family='Bahnschrift Light',
                                                          hoverlabel=dict(font_family='Bahnschrift Light')
                                                          )
    
    return fig_bar, fig_scattermap, fig_sunburst

@app.callback(
    Output('longest_game', 'children'),
    Output('shortest_game', 'children'),
    Output('most_tournaments_won', 'children'),
    Output('most_frequent_matchup', 'children'),
    Output('most_break_points_saved', 'children'),
    Output('most_aces', 'children'),
    Output('radar_chart', 'figure'),
    Output('radar_legend', 'children'),

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
    most_tournaments_won_str=str(tournaments_tournament['singles_winner_name'].value_counts()[0])
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
                                        text = [winner_for_text.index[i] + ': ' + str(winner_for_text['score'][i]) for i in range(len(winner_for_text))],
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
                            margin = dict(l = 80, r = 80, t = 20, b = 20),
                            showlegend = False,
                            template = 'plotly_dark',
                            plot_bgcolor = 'rgba(0, 0, 0, 0)',
                            paper_bgcolor = 'rgba(0, 0, 0, 0)',
                            font_color = 'black',
                            font_size = 15,
                            font_family='Bahnschrift Light',
                            hoverlabel=dict(font_family='Bahnschrift Light')
                            )
    
    ############---RADAR LEGEND----##############################################################################################
    legend='Statistics for the ' + str(year) + ' final match of ' + str(tournament_name) + ', between ' + str(all_match_final['winner_name'].iloc[0]) + ' and ' + str(all_match_final['loser_name'].iloc[0])

    return longest_game_str, shortest_game_str, most_tournaments_won_str, most_frequent_matchup_str, most_break_points_saved_str, most_aces_str, fig_radar, legend


if __name__ == '__main__':
    app.run_server(debug=True)