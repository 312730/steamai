import pandas as pd
import numpy as np
import pickle
from tensorflow import keras


train = pd.read_csv('train-plays.csv').drop(columns=["norm_amount", "amount"])
test = pd.read_csv("test-plays.csv").drop(columns=["norm_amount", "amount"])
game_coding = pd.read_csv("game-coding.csv")

nepochs = 50
early_stop = keras.callbacks.EarlyStopping(monitor='binary_accuracy', min_delta=0.0001, patience=10)


filename = 'final_model.sav'
ncf = pickle.load(open('final_model.sav', errors = 'ignore'))

def recommend_game (uid, model, n=10):
    uid_array = np.repeat(uid, game_coding.game_id.size)
    recs = np.ndarray.flatten(model.predict([uid_array, game_coding.game_id, uid_array, game_coding.game_id]))
    recs_df = pd.DataFrame({'game_id':game_coding.game_id, 'rec_confidence':recs})
    return set(recs_df.sort_values(by='rec_confidence', ascending=False).head(10).game_id)

test['recommended'] = np.vectorize(recommend_game)(test.user_id.unique(), ncf)
test['in_recommendations'] = np.vectorize(lambda gid, recs: 1 if gid in recs else 0)(test.game_id, test.recommended)
test.in_recommendations.sum() / test.in_recommendations.size

games_recommended = set()
np.vectorize(lambda curr_games, total: total.update(curr_games))(test.recommended, games_recommended)
print(recommend_game(30, ncf))
