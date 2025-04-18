import matplotlib.pyplot as plt

train_l1_4 = []
train_psnr_4 = []
test_l1_4 = []
test_psnr_4 = []
duration_4 = []

train_l1_24 = []
train_psnr_24 = []
test_l1_24 = []
test_psnr_24 = []
duration_24 = []

with open("./rpd_graphs_24/4_results.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        d = line.split()
        duration_4.append(float(d[0]))
        train_l1_4.append(float(d[1]))
        train_psnr_4.append(float(d[2]))
        test_l1_4.append(float(d[3]))
        test_psnr_4.append(float(d[4]))


with open("./rpd_graphs_24/24_results.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        d = line.split()
        duration_24.append(float(d[0]))
        train_l1_24.append(float(d[1]))
        train_psnr_24.append(float(d[2]))
        test_l1_24.append(float(d[3]))
        test_psnr_24.append(float(d[4]))


plt.plot(train_l1_4, label='4 views train')
plt.plot(train_l1_24, label='24 views train')
plt.plot(test_l1_4, label='4 views test')
plt.plot(test_l1_24, label='24 views test')
plt.xlabel('Frame')
plt.title('L1')
plt.legend()
plt.savefig('l1.png')
plt.close()

plt.plot(train_psnr_4, label='4 views train')
plt.plot(train_psnr_24, label='24 views train')
plt.plot(test_psnr_4, label='4 views test')
plt.plot(test_psnr_24, label='24 views test')
plt.xlabel('Frame')
plt.title('PSNR')
plt.legend()
plt.savefig('psnr.png')
plt.close()
plt.plot(duration_4, label='4 views')
plt.plot(duration_24, label='24 views')
plt.xlabel('Frame')
plt.ylabel('Seconds')
plt.title('Training Time')
plt.legend()
plt.savefig('duration.png')
plt.close()