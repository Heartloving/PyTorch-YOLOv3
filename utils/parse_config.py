

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions
     输入: 配置文件路径
    返回值: 列表对象,其中每一个元素为一个字典类型对应于一个要建立的神经网络模块（层）
    """
    # 加载文件并过滤掉文本中多余内容
    file = open(path, 'r')
    lines = file.read().split('\n')  # store the lines in a list等价于readlines
    lines = [x for x in lines if x and not x.startswith('#')]   # 去掉空行并且去掉以#开头的注释
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces去掉左右两边的空格
    # cfg文件中的每个块用[]括起来最后组成一个列表，一个block存储一个块的内容，即每个层用一个字典block存储。
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block # 这是cfg文件中一个层(块)的开始
            module_defs.append({})# 在列表最后加入一个空的字典结构
            module_defs[-1]['type'] = line[1:-1].rstrip()# 把cfg的[]中的块名作为新建字典中键type的值
            if module_defs[-1]['type'] == 'convolutional':#如果该块是卷积层，新建一个BN层赋值为0
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")#按等号分割
            value = value.strip()#value去掉左右空格
            module_defs[-1][key.rstrip()] = value.strip()#左边是key(去掉右空格)，右边是value(去掉左右空格)，形成一个block字典的键值对

    # 配置文件定义了6种不同type
    # 'net': 相当于超参数,网络全局配置的相关参数
    # {'convolutional', 'net', 'route', 'shortcut', 'upsample', 'yolo'}
    return module_defs #输出列表，列表中每个字典是网络中的一个块



def parse_data_config(path):
    """Parses the data configuration file
    输入：数据文件路径
    输出：一个包含数据路径等信息的字典
    """
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()#按行读入
    for line in lines:
        line = line.strip()#去掉空格
        if line == '' or line.startswith('#'):#去掉注释和空行
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()#构成字典键值对
    return options#输出字典
