import matplotlib.pyplot as plt

# Dữ liệu của bạn
data = {1: 0.313, 2: 0.412, 3: 0.488, 4: 0.539, 5: 0.578, 6: 0.611, 7: 0.64, 8: 0.661, 9: 0.687, 10: 0.704, 11: 0.719, 12: 0.733, 13: 0.748, 14: 0.759, 15: 0.77, 16: 0.781, 17: 0.79, 18: 0.797, 19: 0.804, 20: 0.808}
x = list(data.keys())    # Lấy keys: [1, 3]
y = list(data.values())  # Lấy values: [2, 4]

# Tạo figure và axes
fig, ax1 = plt.subplots()

# Vẽ trục Y đầu tiên
ax1.plot(x, y, 'b-o', label='Dữ liệu')  # 'b-o' là đường xanh có điểm tròn
ax1.set_xlabel('Trục X')
ax1.set_ylabel('Trục Y', color='b')
ax1.tick_params('y', colors='b')
ax1.grid(True)  # Thêm lưới cho dễ nhìn

# Thêm tiêu đề
#plt.title('Đồ thị từ tập {1: 2, 3: 4}')


# Thêm legend
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

# Điều chỉnh layout và hiển thị
plt.tight_layout()
plt.show()