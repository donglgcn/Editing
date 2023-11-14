import torch
import numpy as np
import matplotlib.pyplot as plt

poi_score = torch.load("poi_score_scaleup.pt")
clean_score = torch.load("clean_score_scaleup.pt")

plt.hist(poi_score, bins=10, alpha=0.5, label='poison')
plt.hist(clean_score, bins=10, alpha=0.5, label='clean')
plt.legend(loc='upper right')
plt.show()


# poi_score = torch.load("poi_score_strip.pt")
# clean_score = torch.load("clean_score_strip.pt")
#
# plt.hist(poi_score, bins=10, alpha=0.5, label='poison')
# plt.hist(clean_score, bins=10, alpha=0.5, label='clean')
# plt.legend(loc='upper right')
# plt.show()