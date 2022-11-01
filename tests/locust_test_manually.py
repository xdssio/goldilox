from locust import HttpUser, TaskSet, task, between


class IrisPredict(TaskSet):
    @task
    def predict(self):
        request_body = [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        ]
        self.client.post('/inference', json=request_body)


class IrisLoadTest(HttpUser):
    tasks = [IrisPredict]
    host = 'http://127.0.0.1:5000'
    # host = 'http://127.0.0.1:8080' # nginx on docker
    stop_timeout = 20
    wait_time = between(1, 5)

# locust -f ./tests/locust_test_manually.py
