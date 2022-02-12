import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import linear_kernel

csv = pd.read_csv('video_games.csv')

tfidf = TfidfVectorizer(stop_words="english")
csv["Metadata.Genres"].isnull().sum() 
csv["Metadata.Genres"] = csv["Metadata.Genres"].fillna(" ") 

def _():
    # print(csv.columns)
    # Title', 'Features.Handheld?', 'Features.Max Players',
    #        'Features.Multiplatform?', 'Features.Online?', 'Metadata.Genres',
    #        'Metadata.Licensed?', 'Metadata.Publishers', 'Metadata.Sequel?',
    #        'Metrics.Review Score', 'Metrics.Sales', 'Metrics.Used Price',
    #        'Release.Console', 'Release.Rating', 'Release.Re-release?',
    #        'Release.Year', 'Length.All PlayStyles.Average',
    #        'Length.All PlayStyles.Leisure', 'Length.All PlayStyles.Median',
    #        'Length.All PlayStyles.Polled', 'Length.All PlayStyles.Rushed',
    #        'Length.Completionists.Average', 'Length.Completionists.Leisure',
    #        'Length.Completionists.Median', 'Length.Completionists.Polled',
    #        'Length.Completionists.Rushed', 'Length.Main + Extras.Average',
    #        'Length.Main + Extras.Leisure', 'Length.Main + Extras.Median',
    #        'Length.Main + Extras.Polled', 'Length.Main + Extras.Rushed',
    #        'Length.Main Story.Average', 'Length.Main Story.Leisure',
    #        'Length.Main Story.Median', 'Length.Main Story.Polled',
    #        'Length.Main Story.Rushed'],
    pass

# print(csv['Metadata.Genres'])

tfidf_matrix = tfidf.fit_transform(csv['Metadata.Genres']) 
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)


game_index = pd.Series(csv.index,index=csv['Title'])

# print(csv.loc[454, 'Title'])
# print(csv)

def get_game_recomendations(name: str, topN):
    # game_id = game_index[name][0] if game_index[name].size != 1 else game_index[name]
    game_id = game_index[name]
    
    cosine_scores = list(enumerate(cosine_sim_matrix[game_id]))
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    cosine_scores_top = cosine_scores[0:topN+1]

    game_idx  =  [i[0] for i in cosine_scores_top]
    game_scores =  [i[1] for i in cosine_scores_top]

    game_similar_show = pd.DataFrame(columns=["Title","Metrics.Review Score"])
    game_similar_show["Title"] = csv.loc[game_idx,"Title"]
    game_similar_show["Metrics.Review Score"] = game_scores
    game_similar_show.reset_index(inplace=True)  
    game_similar_show.drop(["index"],axis=1,inplace=True)
    print (game_similar_show)

    print(csv.loc[game_id, 'Title'])
    print(csv.loc[game_id, 'Metrics.Review Score'])
    print(csv.loc[game_id, 'Metadata.Genres'])

    print(csv.loc[game_idx[0], 'Title'])
    print(csv.loc[game_idx[0], 'Metrics.Review Score'])
    print(csv.loc[game_idx[0], 'Metadata.Genres'])


get_game_recomendations("Ridge Racer",topN=15)
