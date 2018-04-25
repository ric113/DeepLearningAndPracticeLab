import pickle

with open('./data/histories_st.pkl', 'rb') as f:
    data = pickle.load(f)

print(data['loss_history'])