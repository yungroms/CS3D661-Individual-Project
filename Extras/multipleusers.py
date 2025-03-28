from locust import HttpUser, task, between
import os

class LoadTest(HttpUser):
    wait_time = between(1, 2)

    @task
    def upload_image(self):
        with open(R"C:\Users\rms11\Desktop\y3_proj\Test\Test_Dataset_2\Cowslip\20250322_163759.jpg", "rb") as img:
            self.client.post("/", files={"file": img})
