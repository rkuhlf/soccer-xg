import sqlite3
import pandas as pd
from datetime import datetime
import re
import numpy as np

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
 
    return conn



connection = create_connection("database.sqlite")

selectors = []
selectors.append("home_team_api_id")
selectors.append("home_team_goal")
selectors.append("shoton")
selectors.append("shotoff")
selectors.append("cross")
selectors.append("corner")
selectors.append("possession")


games = pd.read_sql_query("SELECT " + ", ".join(selectors) + " FROM match", connection).dropna()






def get_possession(string):
	ans = re.search(r"<homepos>(\d+)</homepos>", string)
	if ans:
		return int(ans.group(1))
	return None



games["possession"] = games["possession"].apply(get_possession)

def get_crosses(string, home_id):
	string = string.split("</value>")
	result = 0
	for s in string:
		ans = re.search(r"(<crosses>(\d+)</crosses>)(?=.*?<team>" + str(home_id) + "</team>)", s)
		if ans:
			result += int(ans.group(2))

	return result

games["cross"] = np.vectorize(get_crosses)(games["cross"], games["home_team_api_id"])


def get_corners(string, home_id):
	string = string.split("</value>")
	result = 0
	for s in string:
		ans = re.search(r"(<corners>(\d+)</corners>)(?=.*?<team>" + str(home_id) + "</team>)", s)
		if ans:
			result += int(ans.group(2))

	return result

games["corner"] = np.vectorize(get_corners)(games["corner"], games["home_team_api_id"])

def get_blocked_shots(string, home_id):
	string = string.split("</value>")
	result = 0
	for s in string:
		ans = re.search(r"(<blocked>(\d+)</blocked>)(?=.*?<team>" + str(home_id) + "</team>)", s)
		if ans:
			result += int(ans.group(2))

	return result

def get_unblocked_shots(string, home_id):
	string = string.split("</value>")
	result = 0
	for s in string:
		ans = re.search(r"(<shoton>(\d+)</shoton>)(?=.*?<team>" + str(home_id) + "</team>)", s)
		if ans:
			result += int(ans.group(2))

	return result

games["unblocked"] = np.vectorize(get_blocked_shots)(games["shoton"], games["home_team_api_id"])
games["blocked"] = np.vectorize(get_unblocked_shots)(games["shoton"], games["home_team_api_id"])

def get_off_target(string, home_id):
	string = string.split("</value>")
	result = 0
	for s in string:
		ans = re.search(r"(<shotoff>(\d+)</shotoff>)(?=.*?<team>" + str(home_id) + "</team>)", s)
		if ans:
			result += int(ans.group(2))

	return result

games["shotoff"] = np.vectorize(get_off_target)(games["shotoff"], games["home_team_api_id"])

games = games.dropna()

games.to_pickle("homeTeamStats.pickle")

# feed neural network the shots off target, the blocked shots, the shots on target, the possession, the corners, and the crosses