# Import libraries
import pandas as pd
import numpy as np

# Load json files
players_df = pd.read_json(r'/Users/davidsousa/Documents/SportsDS/datasets/players.json')
teams_df = pd.read_json(r'/Users/davidsousa/Documents/SportsDS/datasets/teams.json')
events_PL = pd.read_json('/Users/davidsousa/Documents/SportsDS/datasets/events/events_England.json')
events_laliga = pd.read_json('/Users/davidsousa/Documents/SportsDS/datasets/events/events_Spain.json')
events_ligue1 = pd.read_json('/Users/davidsousa/Documents/SportsDS/datasets/events/events_France.json')
events_bundesliga = pd.read_json('/Users/davidsousa/Documents/SportsDS/datasets/events/events_Germany.json')
events_serieA = pd.read_json('/Users/davidsousa/Documents/SportsDS/datasets/events/events_Italy.json')

temp = players_df[[ 'wyId', 'firstName', 'lastName', 'weight', 'height', 'birthDate', 'foot', 'currentTeamId', 'passportArea','birthArea']]
# Merge datasets
players_df = players_df.merge(teams_df, how='inner', left_on='currentTeamId', right_on='wyId')

# Data preprocessing
# Create variables
players_df['Nationality'] = np.nan
for i in range(0, len(players_df)):
    players_df['Nationality'][i] = players_df['birthArea'][i].get('name')

players_df['Position'] = np.nan
for i in range(0, len(players_df)):
    players_df['Position'][i] = players_df['role'][i].get('name')

players_df['LeagueNation'] = np.nan
for i in range(0, len(players_df)):
    players_df['LeagueNation'][i] = players_df['area'][i].get('name')

players_df['League'] = np.nan
players_df.loc[(players_df['LeagueNation'] == 'France') | (players_df['LeagueNation'] == 'Monaco'), 'League'] = 'Ligue 1'
players_df.loc[players_df['LeagueNation'] == 'Germany', 'League'] = 'Bundesliga'
players_df.loc[players_df['LeagueNation'] == 'Spain', 'League'] = 'La Liga'
players_df.loc[(players_df['LeagueNation'] == 'England') | (players_df['LeagueNation'] == 'Wales'), 'League'] = 'Premier League'
players_df.loc[players_df['LeagueNation'] == 'Italy', 'League'] = 'Serie A'

players_df = players_df[['wyId_x', 'shortName', 'weight', 'height', 'Position', 'birthDate', 'foot', 'Nationality', 'name', 'League']]
players_df.columns = ['PlayerID', 'PlayerName', 'Weight', 'Height', 'Position', 'BirthDate', 'Foot', 'Nationality', 'TeamName', 'League']

def preprocessing(df, league):

    # Total number of matches
    total_matches = df.groupby('playerId')['matchId'].nunique().reset_index()
    total_matches.columns = ['playerId', 'total_matches']

    # Total number of passes
    total_passes = pd.DataFrame(df.loc[df['eventName'] == 'Pass']['playerId'].value_counts().reset_index())
    total_passes.columns = ['playerId', 'total_passes']

    # Total number of smart passes
    total_smart_passes = pd.DataFrame(df.loc[(df['eventName'] == 'Pass') &
                                                    (df['subEventName'] == 'Smart pass')]['playerId'].value_counts().reset_index())
    total_smart_passes.columns = ['playerId', 'total_smart_passes']

    # Total number of crosses
    total_crosses = pd.DataFrame(df.loc[(df['eventName'] == 'Pass') &
                                                    (df['subEventName'] == 'Cross')]['playerId'].value_counts().reset_index())
    total_crosses.columns = ['playerId', 'total_crosses']

    # Total number of shots
    total_shots = pd.DataFrame(df.loc[df['eventName'] == 'Shot']['playerId'].value_counts().reset_index())
    total_shots.columns = ['playerId', 'total_shots']

    # Total number of clearances
    total_clearances = pd.DataFrame(df.loc[(df['eventName'] == 'Others on the ball') &
                                                    (df['subEventName'] == 'Clearance')]['playerId'].value_counts().reset_index())
    total_clearances.columns = ['playerId', 'total_clearances']

    # Total number of sprints
    total_sprints = pd.DataFrame(df.loc[(df['eventName'] == 'Others on the ball') &
                                                    (df['subEventName'] == 'Acceleration')]['playerId'].value_counts().reset_index())
    total_sprints.columns = ['playerId', 'total_sprints']

    # Total number of fouls
    total_fouls = pd.DataFrame(df.loc[df['eventName'] == 'Foul']['playerId'].value_counts().reset_index())
    total_fouls.columns = ['playerId', 'total_fouls_committed']

    # Total number of duels
    total_duels = pd.DataFrame(df.loc[df['eventName'] == 'Duel']['playerId'].value_counts().reset_index())
    total_duels.columns = ['playerId', 'total_duels']

    # Total number of duels won
    if league == 'Serie A':
        df.drop(df.tail(4).index, inplace=True) # the last row do not correspond to the end of the match

    df['team_won_duel'] = np.nan
    for i in range(0, len(df)):
        if df['eventName'][i] == 'Duel':
            for j in range(1, len(df)):
                if (df['eventName'][i+j] != 'Duel') & (df['eventName'][i+j] != 'Foul'):
                    df['team_won_duel'][i] = df['teamId'][i+j]
                    break
                elif (df['eventName'][i+j] != 'Duel') & (df['eventName'][i+j] == 'Foul'):
                    df['team_won_duel'][i] = df['teamId'][i+j+1]
                    break
                else:
                    continue

    df['duel_won'] = 0
    df.loc[df['team_won_duel'] == df['teamId'], 'duel_won'] = 1

    total_duels_won = pd.DataFrame(df.loc[df['duel_won'] == 1]['playerId'].value_counts().reset_index())
    total_duels_won.columns = ['playerId', 'total_duels_won']

    # Total number of complete passes
    if league == 'Ligue 1':
        df.drop(df.tail(3).index, inplace=True) # the last row do not correspond to the end of the match

    df['complete_pass'] = np.nan
    for i in range(0, len(df)):

        if df['eventName'][i] == 'Pass':

            if (df['eventName'][i+1] == 'Pass') | (df['eventName'][i+1] == 'Shot') | \
                    (df['subEventName'][i+1] == 'Touch') | (df['subEventName'][i+1] == 'Acceleration') | \
                    (df['eventName'][i + 1] == 'Free Kick'):
                if df['teamId'][i] == df['teamId'][i+1]:
                    df['complete_pass'][i] = 1
                else:
                    df['complete_pass'][i] = 0

            elif (df['subEventName'][i+1] == 'Clearance') | (df['eventName'][i+1] == 'Offside') | \
                    (df['eventName'][i+1] == 'Interruption') | (df['eventName'][i+1] == 'Goalkeeper leaving line') | \
                    (df['eventName'][i+1] == 'Save attempt'):
                df['complete_pass'][i] = 0

            elif df['eventName'][i+1] == 'Foul':
                if df['teamId'][i] == df['teamId'][i+1]:
                    df['complete_pass'][i] = 0
                else:
                    df['complete_pass'][i] = 1

            elif df['eventName'][i+1] == 'Duel':
                if df['teamId'][i] == df['team_won_duel'][i+1]:
                    df['complete_pass'][i] = 1
                else:
                    df['complete_pass'][i] = 0

    if df.loc[df['eventName'] == 'Pass']['complete_pass'].isnull().sum() > 0:
        print('Warning: there are unseen cases of complete passes!')

    total_complete_passes = pd.DataFrame(df.loc[df['complete_pass'] == 1]['playerId'].value_counts().reset_index())
    total_complete_passes.columns = ['playerId', 'total_complete_passes']

    # Total number of interceptions
    df['interception'] = np.nan
    for i in range(1, len(df)):
        if (df['subEventName'][i] == 'Touch') & (df['eventName'][i-1] == 'Pass') & (df['teamId'][i] != df['teamId'][i-1]):
            df['interception'][i] = 1

    total_interceptions = pd.DataFrame(df.loc[df['interception'] == 1]['playerId'].value_counts().reset_index())
    total_interceptions.columns = ['playerId', 'total_interceptions']

    # Total number of blocks
    df['block'] = np.nan
    for i in range(1, len(df)):
        if (df['subEventName'][i] == 'Touch') & (df['eventName'][i-1] == 'Shot') & (df['teamId'][i] != df['teamId'][i-1]):
            df['block'][i] = 1

    total_blocks = pd.DataFrame(df.loc[df['block'] == 1]['playerId'].value_counts().reset_index())
    total_blocks.columns = ['playerId', 'total_blocks']

    # Merge all measures into a single df
    list_measures = [total_clearances, total_duels, total_duels_won, total_fouls, total_matches, total_interceptions, total_blocks,
                     total_passes, total_smart_passes, total_shots, total_sprints, total_crosses, total_complete_passes]
    measures = list_measures[0]
    for df_ in list_measures[1:]:
        measures = measures.merge(df_, on='playerId', how='outer')

    # Remove non-player (id=0)
    measures = measures.loc[measures['playerId'] != 0]

    # Rationing the measures
    measures['clearances_per_game'] = measures['total_clearances'] / measures['total_matches']
    measures['interceptions_per_game'] = measures['total_interceptions'] / measures['total_matches']
    measures['blocks_per_game'] = measures['total_blocks'] / measures['total_matches']
    measures['duels_per_game'] = measures['total_duels'] / measures['total_matches']
    measures['duelswon_per_game'] = measures['total_duels_won'] / measures['total_matches']
    measures['%duelswon_per_game'] = measures['duelswon_per_game'] * 100 / measures['duels_per_game']
    measures['fouls_per_game'] = measures['total_fouls_committed'] / measures['total_matches']
    measures['passes_per_game'] = measures['total_passes'] / measures['total_matches']
    measures['completepasses_per_game'] = measures['total_complete_passes'] / measures['total_matches']
    measures['%passucessrate_per_game'] = measures['completepasses_per_game'] * 100 / measures['passes_per_game']
    measures['smartpasses_per_game'] = measures['total_smart_passes'] / measures['total_matches']
    measures['shots_per_game'] = measures['total_shots'] / measures['total_matches']
    measures['crosses_per_game'] = measures['total_crosses'] / measures['total_matches']
    measures['sprints_per_game'] = measures['total_sprints'] / measures['total_matches']

    measures.drop(columns=['total_clearances', 'total_duels', 'total_duels_won', 'total_fouls_committed',
                           'total_passes', 'total_smart_passes', 'total_crosses', 'total_interceptions', 'total_blocks',
                           'total_sprints', 'total_shots', 'total_complete_passes'], inplace=True)

    # Replacing missing values with 0
    measures.fillna(0, inplace=True)

    # Join measures and dimensions into a single dataframe
    new_df = players_df.merge(measures, how='inner', left_on='PlayerID', right_on='playerId')

    # Exclude players that left the League during the winter transfer window
    new_df = new_df.loc[new_df['League'] == league]

    # Drop duplicate column Player ID
    new_df.drop(columns='playerId', inplace=True)

    return new_df

PremierLeague_df = preprocessing(events_PL, 'Premier League')
LaLiga_df = preprocessing(events_laliga, 'La Liga')
Ligue1_df = preprocessing(events_ligue1, 'Ligue 1')
Bundesliga_df = preprocessing(events_bundesliga, 'Bundesliga')
SerieA_df = preprocessing(events_serieA, 'Serie A')

# Concatenate all the leagues
df = pd.concat([PremierLeague_df, LaLiga_df, Ligue1_df, Bundesliga_df, SerieA_df])

# Set Player ID to index
df.set_index('PlayerID', inplace=True)

# Get insights from the data
info = df.describe()

# Filter by midfielders
midfielders_df = df.loc[df['Position'] == 'Midfielder']

# Export datasets to excel file
df.to_excel('/Users/davidsousa/Documents/SportsDS/datasets/players.xlsx')
midfielders_df.to_excel('/Users/davidsousa/Documents/SportsDS/datasets/midfielders.xlsx')