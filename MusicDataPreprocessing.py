
import pandas as pd


#Initial reading in of the data from online sources (I saved the files to a CSV on my local drive for speed)
triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt' #user data
songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv' #song data

#Store the files in tables/csv using pd and save to hard drive (these won't be used but its good for damage control)
song_df_1 = pd.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']
#song_df_1.to_csv('user_datas.csv')

song_df_2 =  pd.read_csv(songs_metadata_file)
#song_df_2.to_csv('song_datas.csv')

#Merge the tables, remove duplicates and save to hard drive
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

#Do some more data cleaning and transformation
song_df['song'] = song_df['title'] + ' - ' + song_df['artist_name']
song_df = song_df.drop(['title', 'artist_name'], axis = 1)
song_df = song_df[['user_id','song_id', 'song', 'release', 'year']]

#create a new table with aggregated features 'listen count' and 'percentage' for ranking songs in the builder
song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])

#Save the cleaned song-user table and the newly created table
#song_df.to_csv('user_song_data.csv')
#song_grouped.to_csv('song_grouped.csv')


