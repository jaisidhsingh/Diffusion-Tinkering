import pickle


with open("./masks/info.pkl", "rb") as f:
    data = pickle.load(f)

print(data)