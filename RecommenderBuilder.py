#Data managment
import pandas as pd

#Split data into train and test
from sklearn.model_selection import train_test_split

#Machine Learning Models
import Recommenders

#Error evaluation
import time
import Evaluation as Evaluation

#Error visualisation
import pylab as pl

#Method to generate precision and recall curve
def plot_precision_recall(m1_precision_list, m1_recall_list, m1_label,
                          m2_precision_list, m2_recall_list, m2_label):
    pl.clf()
    pl.plot(m1_recall_list, m1_precision_list, label=m1_label)
    pl.plot(m2_recall_list, m2_precision_list, label=m2_label)
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 0.20])
    pl.xlim([0.0, 0.20])
    pl.title('Precision-Recall curve')
    #pl.legend(loc="upper right")
    pl.legend(loc=9, bbox_to_anchor=(0.5, 0.2))
    pl.show()



#Load preprocessed data (song and user)
song_df = pd.read_csv('user_song_data.csv')
song_grouped = pd.read_csv('song_grouped.csv')

#Input Parameters for use in recommender models
subset_size = int(input('Desired subset size (optimum: approx 6000 - 12000): '))
user_index = int(input('Desired user index (1 - 10): '))
top_hits = int(input('How many top hits do you want to see (up to 10)? '))

#Define data subset 
song_df = song_df.iloc[:subset_size, :]

#initialise users (inputs) and songs (outputs)
users = song_df['user_id'].unique()
songs = song_df['song'].unique()

#Split song_df dataframe 
train_data, test_data = train_test_split(song_df, test_size = 0.2, random_state = 42)

#Initialise popularity recommender and item-similarity recommender with training data
pm = Recommenders.popularity_recommender_py()
pm.create(train_data, 'user_id', 'song')

is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')


#Make recommendations based on user history using both models

#Non-personilised recommendations 
user_id = users[user_index]

popular_recommendations = pm.recommend(user_id)
popular_recommendations = popular_recommendations.sort_values('Rank', axis = 0)
print("")
print (f"Most popular songs:\n{popular_recommendations[:top_hits].drop(['user_id', 'score', 'Rank'], axis = 1)}")
print("")
print("")


#Personalised recommendations
user_items = is_model.get_user_items(user_id) #Items used to train this particular useres prediction
personal_recommendations = is_model.recommend(user_id)
print("------------------------------------------------------------------------------------")
print(f"Training data songs for the user id: {user_id}")
print("")
for user_item in user_items:
    print(user_item)
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("")
print (f"Recommended songs:\n{personal_recommendations[:top_hits].drop(['user_id', 'score', 'rank'], axis = 1)}")
print("----------------------------------------------------------------------")
print(" \n Starting error evaluation\n ")
#Evaluation
start = time.time()

#Define what percentage of users to use for precision recall calculation
user_sample = 0.05

#Instantiate the precision_recall_calculator class
pr = Evaluation.precision_recall_calculator(test_data,train_data,
                                            pm, is_model)

#Call method to calculate precision and recall values
(pm_avg_precision_list, pm_avg_recall_list,
ism_avg_precision_list, ism_avg_recall_list) = pr.calculate_measures(user_sample)

end = time.time()
print(end - start)

#Visually display the precision and recall quality of both models

print(" \nPlotting precision recall curves.")
plot_precision_recall(pm_avg_precision_list,
                      pm_avg_recall_list,
                      "popularity_model",
                      ism_avg_precision_list,
                      ism_avg_recall_list, "item_similarity_model")