import pandas as pd

# Đọc dữ liệu từ file CSV
df = pd.read_csv("data/countries-of-the-world.csv")
# print(df[df['Country'].str.contains('Vietnam', case=False)])
print(df.head())
# # Đọc dữ liệu từ file txt và tạo DataFrame
# contents = {}
# with open("data/country_detailed_specialties.txt", "r", encoding="utf-8") as f:
#     for line in f:
#         if ":" in line:
#             key, value = line.split(":", 1)
#             print(value)
#             contents[key.strip()] = value.replace("\"", "").strip()

# df_specia = pd.DataFrame(list(contents.values()), columns=["Specialties"],dtype=str)
# df.insert(2, "Specialties", df_specia)

# # # Lưu DataFrame đã chỉnh sửa vào file CSV
# df.to_csv("data/countries_of_the_world_processing.csv", index=False)

# # print(df.head()) # in ra 5 dòng đầu tiên để kiểm tra kết quả.