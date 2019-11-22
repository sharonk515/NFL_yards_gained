import pandas as pd
import numpy as np
from string import punctuation

def rename_elements(df, df_id):
    '''
    This function renames items in the following columns of the DataFrame:
    StadiumType
    Turf
    GameWeather
    Position

    Parameters
    -----------
    df: DataFrame used for training
    df_id: DataFrame for ID purposes

    Returns
    --------
    DataFrame
    '''

    # set stadium types as outdoor or indoor
    stadium_types = {'Outdoor': 'Outdoor',
                     'Indoor': 'Indoor',
                     'Outdoors': 'Outdoor',
                     'Indoors': 'Indoor',
                     'Dome': 'Indoor',
                     'Retractable Roof': 'Indoor',
                     'Open': 'Outdoor',
                     'Retr. Roof-Closed': 'Indoor',
                     'Retr. Roof - Closed': 'Indoor',
                     'Domed, closed': 'Indoor',
                     'Domed, open': 'Outdoor',
                     'Closed Dome': 'Indoor',
                     'Domed': 'Indoor',
                     'Dome, closed': 'Indoor',
                     'Oudoor': 'Outdoor',
                     'Retr. Roof Closed': 'Indoor',
                     'Indoor, Roof Closed': 'Indoor',
                     'Retr. Roof-Open': 'Outdoor',
                     'Bowl': 'Outdoor',
                     'Outddors': 'Outdoor',
                     'Heinz Field': 'Outdoor',
                     'Retr. Roof - Open': 'Outdoor',
                     'Outdoor Retr Roof-Open': 'Outdoor',
                     'Outdor': 'Outdoor',
                     'Ourdoor': 'Outdoor',
                     'Indoor, Open Roof': 'Outdoor',
                     'Outside': 'Outdoor',
                     'Cloudy': 'Outdoor',
                     'Domed, Open': 'Outdoor'}

    df['StadiumType'] = df['StadiumType'].apply(lambda x: stadium_types[x])

    # set turf type as natural or artificial
    grass = {'Grass': 'Natural',
             'Natural Grass': 'Natural',
             'Field Turf': 'Artificial',
             'Artificial': 'Artificial',
             'FieldTurf': 'Artificial',
             'UBU Speed Series-S5-M': 'Artificial',
             'A-Turf Titan': 'Artificial',
             'UBU Sports Speed S5-M': 'Artificial',
             'FieldTurf360': 'Artificial',
             'DD GrassMaster': 'Mix',
             'Twenty-Four/Seven Turf': 'Artificial',
             'SISGrass': 'Mix',
             'FieldTurf 360': 'Artificial',
             'Natural grass': 'Natural',
             'Artifical': 'Artificial',
             'Natural': 'Natural',
             'Field turf': 'Artificial',
             'Naturall Grass': 'Natural',
             'grass': 'Natural',
             'natural grass': 'Natural'}

    df['Turf'] = df['Turf'].apply(lambda x: grass[x])

    # rename weather values to combine them
    weather = {'Controlled Climate': 'Indoor',
               'N/A (Indoors)': 'Indoor',
               'Indoors': 'Indoor',
               'N/A Indoor': 'Indoor',
               'Sunny, highs to upper 80s': 'Sunny',
               'Cloudy, fog started developing in 2nd quarter': 'Cloudy, fog',
               'Rain likely, temps in low 40s.': 'Rain',
               'Cloudy, 50% change of rain': 'Cloudy',
               'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.': 'Cloudy, rain',
               'Cloudy, chance of rain': 'Cloudy, rain',
               'Cloudy, light snow accumulating 1-3"': 'Cloudy, snow',
               'Rain Chance 40%': 'Rain',
               'Rainy': 'Rain',
               'Partly sunny': 'Partly Sunny',
               'Partly cloudy': 'Partly Cloudy',
               'Clear skies': 'Clear Skies',
               'cloudy': 'Cloudy',
               'Heavy lake effect snow': 'Snow',
               'Cloudy and Cool': 'Cloudy and cool',
               'Partly Clouidy': 'Partly Cloudy',
               'Cloudy, Rain': 'Cloudy, rain',
               'Clear and Cool': 'Clear and cool',
               'Clear and Sunny': 'Clear and sunny',
               'Mostly cloudy': 'Mostly Cloudy',
               'Mostly sunny': 'Mostly Sunny',
               '30% Chance of Rain': 'Rain',
               'Partly Cloudly': 'Partly Cloudy',
               'Coudy': 'Cloudy',
               'Mostly coudy': 'Mostly Cloudy',
               'Partly clear': 'Partly Clear',
               'T: 51; H: 55; W: NW 10 mph': 'Fair',
               'Party Cloudy': 'Partly Cloudy',
               'Mostly Coudy': 'Mostly Cloudy'}

    df['GameWeather'] = df['GameWeather'].apply(lambda x: weather[x]
                                                if x in weather.keys()
                                                else x)

    # rename positions to combine them into one
    pos = {'OLB': 'LB',
           'ILB': 'LB',
           'MLB': 'LB',
           'SS': 'S',
           'FS': 'S',
           'SAF': 'S',
           'T': 'OT'}

    df['Position'] = df['Position'].apply(lambda x: pos[x]
                                          if x in pos.keys()
                                          else x)

    # rename cities to match the format of 'city, state'
    cities = {'Orchard Park NY': 'Orchard Park, NY',
              'Chicago. IL': 'Chicago, IL',
              'Houston, Texas': 'Houston, TX',
              'Los Angeles, Calif.': 'Los Angeles, CA',
              'Arlington, Texas': 'Arlington, TX',
              'Baltimore, Md.': 'Baltimore, MD',
              'Charlotte, North Carolina': 'Charlotte, NC',
              'Indianapolis, Ind.': 'Indianapolis, IN',
              'Cincinnati, Ohio': 'Cincinnati, OH',
              'Pittsburgh': 'Pittsburgh, PA',
              'Detroit': 'Detroit, MI',
              'Foxborough, Ma': 'Foxborough, MA',
              'Miami Gardens, Fla.': 'Miami Gardens, FL',
              'Philadelphia, Pa.': 'Philadelphia, PA',
              'London': 'London, England',
              'New Orleans, La.': 'New Orleans, LA',
              'Mexico City': 'Mexico City, Mexico',
              'Baltimore, Maryland': 'Baltimore, MD',
              'Jacksonville, Fl': 'Jacksonville, FL',
              'Jacksonville, Florida': 'Jacksonville, FL',
              'Cleveland,Ohio': 'Cleveland, OH',
              'East Rutherford, N.J.': 'East Rutherford, NJ',
              'E. Rutherford, NJ': 'East Rutherford, NJ',
              'Seattle': 'Seattle, WA',
              'Cleveland Ohio': 'Cleveland, OH',
              'Miami Gardens, FLA': 'Miami Gardens, FL',
              'Cleveland': 'Cleveland, OH',
              'Kansas City,  MO': 'Kansas City, MO',
              'Jacksonville Florida': 'Jacksonville, FL',
              'New Orleans': 'New Orleans, LA',
              'Cleveland, Ohio': 'Cleveland, OH'}

    df_id['Location'] = df_id['Location'].apply(lambda x: cities[x]
                                                if x in cities.keys()
                                                else x)

    return df


def replace_null_cols(df):
    '''
    This function finds null columns and replaces them with appropriate values.
    Categorical columns will be replaced with the most common values.
    Numerical columns will be replaced with the median values.

    Parameters
    -----------
    df: DataFrame

    Returns
    --------
    DataFrame
    '''
    null_cols = []
    for col in df.columns:
        if df[col].isna().any() == True:
            null_cols.append(col)

    for col in null_cols:
        if df[col].dtype == 'O':
            df[col].fillna(df[col].value_counts().index[0],
                           inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    return df


def give_me_WindSpeed(x):
    x = str(x)
    x = x.replace('mph', '').strip()
    if '-' in x:
        x = (int(x.split('-')[0]) + int(x.split('-')[1])) / 2
    try:
        return float(x)
    except:
        return -99


def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans


def organize_team_abbrs(df):
    '''
    This function reorganizes team abbreviations,
    making sure abbreviations are the same across various columns.

    Parameters
    -----------
    df: DataFrame

    Returns
    --------
    DataFrame
    '''
    df['TeamAbbr'] = df.apply(lambda row: row['HomeTeamAbbr']
                              if row['Team'] == 'home'
                              else row['VisitorTeamAbbr'], axis=1)
    df['OppTeamAbbr'] = df.apply(lambda row: row['VisitorTeamAbbr']
                                 if row['Team'] == 'home'
                                 else row['HomeTeamAbbr'], axis=1)
    df.drop(['HomeTeamAbbr', 'VisitorTeamAbbr'], axis=1, inplace=True)

    diff_abbr = {}
    for x, y in zip(sorted(df['TeamAbbr'].unique()),
                    sorted(df['PossessionTeam'].unique())):
        if x != y:
            diff_abbr[x]: y

    for abb in df['PossessionTeam'].unique():
        diff_abbr[abb] = abb

    df['PossessionTeam'] = df['PossessionTeam'].map(diff_abbr)
    df['TeamAbbr'] = df['TeamAbbr'].map(diff_abbr)
    df['OppTeamAbbr'] = df['OppTeamAbbr'].map(diff_abbr)

    return df


def clean_WindDirection(txt):
    if pd.isna(txt):
        return np.nan
    txt = txt.lower()
    txt = ''.join([c for c in txt if c not in punctuation])
    txt = txt.replace('from', '')
    txt = txt.replace(' ', '')
    txt = txt.replace('north', 'n')
    txt = txt.replace('south', 's')
    txt = txt.replace('west', 'w')
    txt = txt.replace('east', 'e')
    txt = txt.replace('13', 'wnw')
    txt = txt.replace('1', 'nne')
    txt = txt.replace('8', 's')
    txt = txt.replace('calm', 'ne') # most common value
    return txt


def new_orientation(angle, play_direction):
    if play_direction == 0:
        new_angle = 360.0 - angle
        if new_angle == 360.0:
            new_angle = 0.0
        return new_angle
    else:
        return angle


def assign_yard_class(x):
    if x < 0:
        return "< 0"
    elif (x >= 0 and x <= 1):
        return "0-1"
    elif (x > 1 and x <= 3):
        return "1-3"
    elif (x > 3 and x <= 6):
        return "3-6"
    else:
        return "> 6"
