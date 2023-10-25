import os
import yaml
import os.path as osp


def get_config(dir):
    # add direction join function when parse the yaml file
    def join(loader, node):
        seq = loader.construct_sequence(node)#将路径和文件名放到一起
        return os.path.sep.join(seq)     #将路径和文件名中间加上  \  然后连接起来

    # add string concatenation function when parse the yaml file
    def concat(loader, node):
        seq = loader.construct_sequence(node)
        seq = [str(tmp) for tmp in seq]
        return ''.join(seq)           #连接操作，中间不加入任何符合

    yaml.add_constructor('!join', join)
    yaml.add_constructor('!concat', concat)
    with open(dir, 'r', encoding= 'utf-8') as f:
        cfg = yaml.load(f,Loader=yaml.FullLoader)

    check_dirs(cfg)
    return cfg


def check_dir(folder, mk_dir=True):
    if not osp.exists(folder):
        if mk_dir:
            print(f'making direction {folder}!')
            os.mkdir(folder)
        else:
            raise Exception(f'Not exist direction {folder}')


def check_dirs(cfg):
    check_dir(cfg['data_root'], mk_dir=False)#检查yaml文件中各个目录是否可用
    #check_dir(cfg['result_root'])
    #check_dir(cfg['ckpt_folder'])
    #check_dir(cfg['result_sub_folder'])
