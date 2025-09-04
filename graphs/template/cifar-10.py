import matplotlib.pyplot as plt

baseline_train = 
baseline_test = 

AA_train = 
AA_test = 

OA_train = 
OA_test = 

comb_train = 
comb_test = 

fig, axes = plt.subplots(2, 2)

axes[0, 0].plot(baseline_train, label='Training Accuracy')
axes[0, 0].plot(baseline_test, label='Test Accuracy')
axes[0, 0].set_title("Baseline")
axes[0, 0].set_xlabel("Epochs")
axes[0, 0].set_ylim(top=100)
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True)

axes[1, 0].plot(AA_train, label='Training Accuracy')
axes[1, 0].plot(AA_test, label='Test Accuracy')
axes[1, 0].set_title("AutoAugment")
axes[1, 0].set_xlabel("Epochs")
axes[1, 0].set_ylim(top=100)
axes[1, 0].legend(loc='lower right')
axes[1, 0].grid(True)

axes[0, 1].plot(OA_train, label='Training Accuracy')
axes[0, 1].plot(OA_test, label='Test Accuracy')
axes[0, 1].set_title("OnlineAugment")
axes[0, 1].set_xlabel("Epochs")
axes[0, 1].set_ylim(top=100)
axes[0, 1].legend(loc='lower right')
axes[0, 1].grid(True)

axes[1, 1].plot(comb_train, label='Training Accuracy')
axes[1, 1].plot(comb_test, label='Test Accuracy')
axes[1, 1].set_title("AutoAugment + OnlineAugment")
axes[1, 1].set_xlabel("Epochs")
axes[1, 1].set_ylim(top=100)
axes[1, 1].legend(loc='lower right')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()