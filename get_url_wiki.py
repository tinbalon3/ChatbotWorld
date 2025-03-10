import requests
from bs4 import BeautifulSoup

def get_country_flag_links(url, output_file):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Tìm tất cả thẻ img với alt="Flag of ..."
    images = soup.find_all('img', alt=lambda x: x and x.startswith('Flag of'))

    with open(output_file, 'w', encoding='utf-8') as f:
        for img in images:
            src = img.get('src')
            if src:
                # Nếu đường dẫn là tương đối, chuyển thành tuyệt đối
                if src.startswith('/'):
                    src = requests.compat.urljoin(url, src)
                f.write(src + '\n')

# Sử dụng hàm với ví dụ cụ thể:
url = "https://www.countryflags.com/"
output_file = "data/texts/country_flag_test.txt"

get_country_flag_links(url, output_file)
