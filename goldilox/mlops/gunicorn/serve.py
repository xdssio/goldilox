import os
from distutils.util import strtobool

import goldilox

if __name__ == '__main__':
    nginx = strtobool(os.getenv('NGINX'))
    nginx_config = os.getenv('NGING_CONFIG'), 'nginx.conf'
    root_path = os.getenv('ROOT_PATH')
    path = os.getenv('PATH', 'pipeline.pkl')
    server = goldilox.app.GoldiloxServer(path=path, root_path=root_path, nginx_config=nginx_config, options=options)
    server.serve(nginx=nginx)
