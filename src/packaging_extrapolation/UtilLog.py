import os
import re

# 修改gif基组关键字，chk_sym：原chk标识,key_word:原key words处标识
import shutil


def update_card(source_folder, target_folder, chk_sym, key_word, new_chk_sym, new_key_word=None):
    if new_chk_sym == 'avdz':
        new_key_word = 'aug-cc-pvdz'
    elif new_chk_sym == 'avtz':
        new_key_word = 'aug-cc-pvtz'
    elif new_chk_sym == 'avqz':
        new_key_word = 'aug-cc-pvqz'
    elif new_chk_sym == 'av5z':
        new_key_word = 'aug-cc-pv5z'
    elif new_chk_sym == 'av6z':
        new_key_word = 'aug-cc-pv6z'

    # 创建目标文件夹
    os.makedirs(target_folder, exist_ok=True)

    # 遍历源文件夹中的文件
    for filename in os.listdir(source_folder):
        source_file_path = os.path.join(source_folder, filename)
        target_file_path = os.path.join(target_folder, filename)

        with open(source_file_path, 'r') as source_file:
            lines = source_file.readlines()

        modified_lines = []
        for line in lines:
            modified_line = line.replace(chk_sym, new_chk_sym).replace(key_word, new_key_word)
            modified_lines.append(modified_line)

        with open(target_file_path, 'w') as target_file:
            target_file.writelines(modified_lines)


# 修改文件名，替换文件名中的file_word为new_file_word
def update_filename(source_folder, file_word, new_file_word):
    # 获取文件夹中的所有文件名
    file_names = os.listdir(source_folder)

    # 遍历文件名列表并修改文件名
    for file_name in file_names:
        # 判断文件名中是否包含'avdz'
        if file_word in file_name:
            # 构造新的文件名
            new_file_name = file_name.replace(file_word, new_file_word)

            # 构造旧文件路径和新文件路径
            old_file_path = os.path.join(source_folder, file_name)
            new_file_path = os.path.join(source_folder, new_file_name)

            # 重命名文件
            os.rename(old_file_path, new_file_path)

            print(f"文件名已修改：{file_name} -> {new_file_name}")

    print("文件名修改完成。")


# 复制文件夹到新的文件夹,并重命名
def copy_file(source_file, target_folder, new_filename):
    # 构造新的文件路径
    target_file_path = os.path.join(target_folder, new_filename)
    shutil.copy(source_file, target_file_path)


# 更改gjf，由上一次波函数作为初猜进行计算
def update_chk(source_gjf, target_gjf, *, new_chk, old_method, new_method,
               old_card, new_card):
    # 遍历文件夹文件
    for filename in os.listdir(source_gjf):
        # 拼接路径
        source_gjf_path = os.path.join(source_gjf, filename)

        # 读取文件
        with open(source_gjf_path, 'r') as source_file:
            lines = source_file.readlines()

        modified_lines = []

        # 遍历文本行
        for i, line in enumerate(lines):
            # 当遍历到%chk这一行时，更改
            if line.startswith("%chk="):
                modified_line = line.replace(old_method, new_method).replace(old_card, new_card)
                modified_lines.append(modified_line)
            # 当遍历到第4行，更改行
            elif line.startswith("#p"):
                modified_line = new_chk
                modified_lines.append(modified_line)
                modified_lines.append('\n')
            else:
                modified_lines.append(line)

            # 判断是否读入到0 1
            if i == 7:
                modified_lines.append('\n')
                break

        # 目标文件绝对路径
        target_gif_path = os.path.join(target_gjf, filename)

        # 将gjf文件写入新的文件夹
        with open(target_gif_path, 'w') as target_file:
            target_file.writelines(modified_lines)

    # 更改文件名
    update_filename(target_gjf, old_method, new_method)
    update_filename(target_gjf, old_card, new_card)


# 文件夹获取hf,mp2,ccsd,ccsd(t)能量
def get_energy_values(source_folder):
    for filename in os.listdir(source_folder):
        # 读取文件内容
        data = get_log_values(filename)


# 获取单个文件能量
def get_log_values(source_file,method_type):
    # 读取文件内容
    with open(source_file, 'r') as file:
        content = file.read()

    # 开始位置
    start_index = content.find('1\\1')
    # 存储数据的列表
    data = []
    # 判断找到指定位置
    if start_index != -1:
        # 从指定位置开始按 `\` 分割内容
        split_content = content[start_index:].split('\\')
        # 遍历分割结果
        for item in split_content:
            # 判断是否遇到结束标记
            if item == '@':
                break
            item = item.replace('\n', '').replace(' ', '')
            # 存储数据
            data.append(item)
    HF = float(get_HF(data))
    MP2 = float(get_MP2(data))
    MP4 = float(get_MP4(data))
    CCSD = float(get_CCSD(data))
    CCSD_T = float(get_CCSD_T(data))
    energy_dict = {'HF': HF, 'MP2': MP2, 'MP4': MP4, 'CCSD': CCSD,
                   'CCSD(T)': CCSD_T}
    return energy_dict.get(method_type)


# 批量更改gjf内存
def update_mem(folder_path, output_folder_path, old_value, new_value):
    # 创建新文件夹
    os.makedirs(output_folder_path, exist_ok=True)

    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        # 拼接文件路径
        file_path = os.path.join(folder_path, filename)
        output_file_path = os.path.join(output_folder_path, filename)

        # 判断是否为文件
        if os.path.isfile(file_path):
            # 读取文件内容
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # 修改等号后面的值
            for i, line in enumerate(lines):
                if line.startswith(old_value):
                    parts = line.split("=")
                    if len(parts) > 1:
                        value = new_value  # 替换为你想要的新值
                        new_line = parts[0] + "=" + value + "\n"
                        lines[i] = new_line

            # 将修改后的内容写入新文件
            with open(output_file_path, 'w') as file:
                file.writelines(lines)

            print(f"文件 {filename} 修改并保存成功！")

    print("所有文件修改并保存完成！")


# 获取单个能量
def get_HF(data):
    for i in range(len(data)):
        item = data[i]
        if 'HF=' in item:
            return re.search(r'-?\d+\.\d+', item).group()
    return ValueError('No HF-energy can get')


def get_MP2(data):
    for i in range(len(data)):
        item = data[i]
        if 'MP2=' in item:
            return re.search(r'-?\d+\.\d+', item).group()
    return ValueError('No MP2-energy can get')


def get_MP4(data):
    for i in range(len(data)):
        item = data[i]
        if 'MP4SDQ=' in item:
            return re.search(r'-?\d+\.\d+', item).group()
    return ValueError('No MP4-energy can get')


def get_CCSD(data):
    for i in range(len(data)):
        item = data[i]
        if 'CCSD=' in item:
            return re.search(r'-?\d+\.\d+', item).group()
    return ValueError('No HF-energy can get')


def get_CCSD_T(data):
    for i in range(len(data)):
        item = data[i]
        if 'CCSD(T)=' in item:
            return re.search(r'-?\d+\.\d+', item).group()
    return ValueError('No CCSD(T)-energy can get')


def update_method(source_gjf, target_gjf, *, new_chk, old_method, new_method,
                  old_card, new_card):
    # 遍历文件夹文件
    for filename in os.listdir(source_gjf):
        # 拼接路径
        source_gjf_path = os.path.join(source_gjf, filename)

        # 读取文件
        with open(source_gjf_path, 'r') as source_file:
            lines = source_file.readlines()

        modified_lines = []

        # 遍历文本行
        for i, line in enumerate(lines):
            # 当遍历到%chk这一行时，更改
            if line.startswith("%chk="):
                modified_line = line.replace(old_method, new_method).replace(old_card, new_card)
                modified_lines.append(modified_line)
            # 当遍历到第4行，更改行
            elif line.startswith("#p"):
                modified_line = new_chk
                modified_lines.append(modified_line)
                modified_lines.append('\n')
            else:
                modified_lines.append(line)

        # 目标文件绝对路径
        target_gif_path = os.path.join(target_gjf, filename)

        # 将gjf文件写入新的文件夹
        with open(target_gif_path, 'w') as target_file:
            target_file.writelines(modified_lines)

    # 更改文件名
    update_filename(target_gjf, old_method, new_method)
    update_filename(target_gjf, old_card, new_card)
