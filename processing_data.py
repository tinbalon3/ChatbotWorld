import pandas as pd

def sort_description_by_countries_and_fill_missing(countries_file, description_file, output_file):
    # 1. Đọc danh sách quốc gia
    with open(countries_file, 'r', encoding='utf-8') as f:
        countries_list = [line.strip() for line in f if line.strip()]

    # 2. Đọc nội dung file description_flag
    #    Tạo một từ điển "quốc gia -> dòng mô tả"
    description_dict = {}
    with open(description_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_content = line.strip()
            # Ở đây ta cần cách xác định tên quốc gia trong dòng, tuỳ vào định dạng thực tế.
            # Ví dụ đơn giản: nếu dòng bắt đầu bằng tên quốc gia, rồi tới mô tả
            # => ta tách bằng dấu ':' hoặc dấu '-' hay một quy ước nào đó.
            
            # Giả sử format: "Vietnam: Đây là mô tả cờ Việt Nam"
            # thì ta có thể tách như sau:
            parts = line_content.split(':', maxsplit=1)
            if len(parts) == 2:
                country_name = parts[0].strip()
                description  = parts[1].strip()
                description_dict[country_name] = line_content
            else:
                # Nếu không đúng format, ta có thể bỏ qua hoặc xử lý riêng
                # Ở đây ta cho qua
                pass

    # 3. Tạo danh sách kết quả sắp xếp theo thứ tự trong countries_list
    sorted_descriptions = []
    for country in countries_list:
        if country in description_dict:
            # Đã có mô tả, lấy dòng gốc
            sorted_descriptions.append(description_dict[country])
        else:
            # Chưa có mô tả -> thêm "tên_quốc_gia : ..."
            sorted_descriptions.append(f"{country} : ...")

    # 4. Ghi kết quả ra file
    with open(output_file, 'w', encoding='utf-8') as f:
        for desc_line in sorted_descriptions:
            f.write(desc_line + '\n')

def extract_links(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    links = []
    for line in lines:
        # Giả sử định dạng mỗi dòng: "Tên quốc gia : link"
        parts = line.split(":", 1)  # Tách dựa trên dấu ":" đầu tiên
        if len(parts) == 2:
            link = parts[1].strip()  # Loại bỏ khoảng trắng dư
            links.append(link)
        else:
            # Nếu dòng không đúng định dạng, có thể bỏ qua hoặc xử lý riêng.
            pass

    with open(output_file, "w", encoding="utf-8") as fout:
        for link in links:
            fout.write(link + "\n")

# Sử dụng hàm:
# extract_links("data/texts/country_flags.txt", "data/texts/country_flags_links.txt")

# Gọi hàm
# sort_description_by_countries_and_fill_missing(
#     countries_file='data/texts/countries_of_the_world.txt',
#     description_file='data/texts/description_flag.txt',
#     output_file='data/texts/description_flag_sorted.txt'
# )

# Hàm sắp xếp lại mô tả theo thứ tự các quốc gia trong country_flag_test.txt
def sort_descriptions_by_country_flag(country_flag_file, description_file, output_file):
    # Đọc danh sách tên quốc gia từ file country_flag_test.txt
    with open(country_flag_file, 'r', encoding='utf-8') as f:
        country_urls = [line.strip() for line in f if line.strip()]

    # Trích xuất tên quốc gia từ URL
    countries_order = []
    for url in country_urls:
        country_name = url.split('/')[4].replace('-', ' ').title()
        countries_order.append(country_name)

    # Đọc nội dung mô tả từ file description_flag_sorted.txt
    with open(description_file, 'r', encoding='utf-8') as f:
        descriptions = f.read().split('\n')

    # Tạo dictionary lưu mô tả theo tên quốc gia
    desc_dict = {}
    current_country = ""
    for line in descriptions:
        if line.strip() == "":
            continue
        if ':' in line:
            current_country = line.split(':')[0].strip()
            desc_dict[current_country] = line
        else:
            desc_dict[current_country] += " " + line.strip()

    # Sắp xếp mô tả theo danh sách country flag
    sorted_desc = []
    for country in countries_order:
        description = desc_dict.get(country, f"{country} : ...")
        sorted_desc.append(description)

    # Ghi mô tả đã sắp xếp ra file output
    with open(output_file, 'w', encoding='utf-8') as f:
        for desc in sorted_desc:
            f.write(desc + "\n")

# Cách dùng
sort_descriptions_by_country_flag(
    country_flag_file="data/texts/country_flag_test.txt",
    description_file="data/texts/description_flag_sorted.txt",
    output_file="data/texts/sorted_descriptions.txt"
)

