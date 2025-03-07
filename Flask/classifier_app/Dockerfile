# Use an official lightweight Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory into the container
COPY . /app

# Install dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 for the Flask app
EXPOSE 8080

# Start the Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
