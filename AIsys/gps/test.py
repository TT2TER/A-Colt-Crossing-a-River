import csv

# 创建一个空列表来存储解析后的GPGGA消息
gpgga_messages = []

# 输入文件名
input_filename = 'gps_data.txt'
# 输出CSV文件名
output_filename = 'parsed_gps_data.csv'

# 打开输入文件并读取数据
with open(input_filename, 'r') as file:
    lines = file.readlines()

# 遍历每一行
for line in lines:
    # 判断行是否包含GPGGA消息
    if line.startswith('$GPGGA'):
        # 使用逗号分割字段
        fields = line.split(',')
        # 提取所需的信息
        time_utc = fields[1]
        latitude = fields[2]
        lat_direction = fields[3]
        longitude = fields[4]
        lon_direction = fields[5]
        position_quality = fields[6]
        satellites_used = fields[7]
        horizontal_accuracy = fields[8]
        altitude = fields[9]
        altitude_units = fields[10]
        # 提取差分时间和差分站ID
        differential_time = fields[11]
        differential_station_id = fields[12]
        # 创建一个包含提取信息的字典
        gpgga_info = {
            "UTC时间": time_utc,
            "纬度": latitude,
            "纬度半球": lat_direction,
            "经度": longitude,
            "经度半球": lon_direction,
            "定位质量指示": position_quality,
            "使用卫星数量": satellites_used,
            "水平精度因子": horizontal_accuracy,
            "天线海拔": altitude,
            "海拔单位": altitude_units,
            "差分时间": differential_time,
            "差分站ID": differential_station_id
        }
        # 将字典添加到列表中
        gpgga_messages.append(gpgga_info)

# 将解析后的数据写入CSV文件
with open(output_filename, 'w', newline='') as csvfile:
    fieldnames = ["UTC时间", "纬度", "纬度半球", "经度", "经度半球", "定位质量指示", "使用卫星数量", "水平精度因子", "天线海拔", "海拔单位", "差分时间", "差分站ID"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # 写入CSV文件的标题行
    writer.writeheader()
    
    # 写入解析后的数据
    for message in gpgga_messages:
        writer.writerow(message)

print(f"数据已成功保存到 {output_filename}")
