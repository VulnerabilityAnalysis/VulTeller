import os
import sys
import re
import pandas as pd
import subprocess
import signal


def analysis(row):
    id_, func = str(row['id']), row['function']
    code_file = 'functions/' + id_ + '.cc'
    with open(code_file, 'w') as fw:
        func = re.sub(r'.*?(\w+\s*\([\w\W+]*\)[\s\n]*\{)', r'void \1', func, 1)
        func = re.sub(r'(\)\s*)\w+(\s*\{)', r'\1 \2', func, 1)
        fw.write("#define __user __attribute__((noderef, address_space(__user)))\n")
        fw.write(func)
    if not os.path.exists(f'bin/{id_}_cpg.bin'):
        os.system(f"joern-parse {code_file} --max-num-def 10000000 --output bin/{id_}_cpg.bin")
    if not os.path.exists(f'prop/{id_}_all.json'):
        start, stop = 2, len(func.splitlines())+1
        cmd_str = 'joern --script parse.sc -p bin="bin/%s_cpg.bin",start=%d,stop=%d,' \
                  'fid=%s' % (id_, start, stop, id_)
        p = subprocess.Popen(cmd_str, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True, close_fds=True,
                             start_new_session=True)
        try:
            msg, err = p.communicate(timeout=300)
            ret_code = p.poll()
            # print(ret_code, msg, err)
            print(str(msg.decode('utf-8')))
        except subprocess.TimeoutExpired:
            p.kill()
            p.terminate()
            os.killpg(p.pid, signal.SIGTERM)


def read_subset(subset):
    data = pd.read_csv(f'../data/{subset}.csv')
    for i, row in data.iterrows():
        if i in range(start, stop):
            analysis(row)


if __name__ == '__main__':
    start, stop = int(sys.argv[1]), int(sys.argv[2])
    read_subset('train')
    read_subset('valid')
    read_subset('test')
