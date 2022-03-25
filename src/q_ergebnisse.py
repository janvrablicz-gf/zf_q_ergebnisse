#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pyreadr
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import torch
import torch.nn as nn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

#%%
result = pyreadr.read_r(
    "N:/ATHE/HELG/Office/Technik/03_Fachbereiche/Prozessentwicklung/02_Datentransparenz/datenlesen/data_ZF/dfFokus.RDS",
    timezone="UTC",
)
# %%
df = result[None]
df = df[df.Bauteil == "HP09120"]
df = df[df.ActRank == 1]
col = ["FormNr", "Bauteil", "MACHINED_PART"]

print(df.shape)
for c in col:
    df = df[df[c] != "nan"]
    df[c] = df[c].astype("string")

#%%
df.loc[:, "FormBauteilMP"] = df.FormNr + df.Bauteil + df.MACHINED_PART
df = df.dropna(axis=0, subset="FormBauteilMP")
df = df.dropna(axis=0, subset=["M1", "M2", "TZ"])
df = df[df["ANFA"] == "0"]
# %%

X_cat = df[["DGM", "BA", "MACHINING_PLANT_LINE", "FormBauteilMP"]]

X_continous = df[["DR", "M1", "M2", "H3", "TZ", "V1", "V2"]]

oh = OneHotEncoder(sparse=False)
X_cat = oh.fit_transform(X_cat)

X = np.concatenate([X_cat, X_continous.values], axis=1)
#%%
y = (df.CATEGORY == "Gut").astype(int)
weights = y.value_counts(normalize=True)
y = y.values
#%%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
)
# %%
model = LogisticRegressionCV(solver="lbfgs", class_weight="balanced", cv=10).fit(
    X_train, y_train
)
# %%
cr = classification_report(y_test, model.predict(X_test))
cm = ConfusionMatrixDisplay(confusion_matrix(y_test, model.predict(X_test)))
print(cr)

#%%
rus = RandomUnderSampler(random_state=0, replacement=True)
X_train_r, y_train_r = rus.fit_resample(X_train, y_train)
# %%
model = RandomForestClassifier().fit(X_train_r, y_train_r)
cr = classification_report(y_test, model.predict(X_test))
print(cr)
# %%
X_train_r, y_train_r = SMOTE().fit_resample(X_train, y_train)
model = RandomForestClassifier().fit(X_train_r, y_train_r)
cr = classification_report(y_test, model.predict(X_test))
cm = ConfusionMatrixDisplay(confusion_matrix(y_test, model.predict(X_test)))
print(cr)
# %%
scaler = StandardScaler()
X_train_r = scaler.fit_transform(X_train_r)
X_test = scaler.transform(X_test)


class Net(nn.Module):
    def __init__(self, input_shape, output_shape) -> None:
        super(Net, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_shape, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 50),
            nn.LeakyReLU(),
            nn.Linear(50, output_shape),
        )

    def forward(self, x):
        return self.model(x)


net = Net(X_train_r.shape[1], 1)
optim = torch.optim.Adam(net.model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()
batch_size = 64
epoch_num = 10
# %%
losses = []
for epoch in range(epoch_num):
    for i in range(0, len(X_train_r), batch_size):
        X_batch, y_batch = [
            torch.tensor(batch).float()
            for batch in [X_train_r[i : i + batch_size], y_train_r[i : i + batch_size]]
        ]
        optim.zero_grad()

        y_pred = net(X_batch)
        loss = loss_fn(y_pred, y_batch.unsqueeze(1))
        loss.backward()
        optim.step()

        # loss output
        losses.append(loss.item())
        # print(f"[{epoch + 1}, {i + 1:5d}] loss: {loss.item()}")

# y_pred = net(X_)
# %%
plt.plot(losses)
# %%
y_pred = net(torch.tensor(X_test).float())  #
y_pred = torch.round(torch.sigmoid(y_pred)).detach().numpy()
cr = classification_report(y_test, y_pred)
cm = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
print(cr)
cm.plot()
# %%
