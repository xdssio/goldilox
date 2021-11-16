from goldilox.app import Server


def main():
    path = 'tests/models/server.pkl'
    options = {'workers': 2,
               'bind': f"localhost:5000",
               'preload': True,
               'worker_class': "uvicorn.workers.UvicornH11Worker"
               }
    # can only create the app after starting redis

    Server(path, options).serve()


if __name__ == "__main__":
    main()
