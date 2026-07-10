# AWS EC2 & ECR Deployment Guide

This guide details the end-to-end setup and deployment steps to host this Flask Machine Learning application on AWS EC2, using Amazon ECR (Elastic Container Registry) and GitHub Actions for automated CI/CD.

---

## 🛠️ Prerequisites & Setup Steps

Follow these steps to configure your AWS infrastructure, setup your EC2 host environment, and establish automated deployment.

### 1. AWS IAM Configuration (Identity & Access Management)

Create an IAM User to allow GitHub Actions to securely build, push, and deploy containers:

1. Navigate to the **IAM Console** in AWS.
2. Select **Users** and click **Create user**.
3. Name the user (e.g., `github-actions-ml-deploy`).
4. Attach the following managed policies directly to the user:
   - `AmazonEC2ContainerRegistryFullAccess`: Required to authenticate with ECR, create/delete repositories, and push/pull Docker images.
   - `AmazonEC2FullAccess` (Optional, or narrow custom policy if EC2 instance status needs to be managed).
5. Proceed to create the user.
6. Open the newly created user's profile, navigate to the **Security credentials** tab, and click **Create access key**.
7. Choose **Command Line Interface (CLI)** or **Other**, save the generated **AWS Access Key ID** and **AWS Secret Access Key**. (Keep these safe, you will need them for GitHub Secrets).

---

### 2. AWS Elastic Container Registry (ECR) Setup

Create a private repository to host your Docker images:

1. Navigate to the **Amazon ECR** console.
2. Click **Create repository**.
3. Choose **Private** visibility settings.
4. Set the repository name (e.g., `math-marks-prediction`).
5. Keep other settings default and click **Create repository**.
6. Copy the **URI** of your repository (formatted as: `<aws_account_id>.dkr.ecr.<region>.amazonaws.com/math-marks-prediction`).

---

### 3. AWS EC2 Instance Setup

Launch and configure your virtual server:

1. Navigate to the **EC2 Console** and click **Launch instance**.
2. **Name:** `ml-prediction-server`.
3. **OS Image:** Select **Ubuntu** (Ubuntu Server 22.04 LTS or 24.04 LTS recommended).
4. **Instance Type:** Select `t2.micro` (free-tier eligible) or `t2.medium` (recommended if running heavy model inference).
5. **Key Pair:** Create or select an existing `.pem` key pair for SSH access.
6. **Network Settings (Security Group):**
   - Allow **SSH** traffic (Port `22`) from your IP.
   - Allow **HTTP** traffic (Port `80`) from Anywhere.
   - Allow **HTTPS** traffic (Port `443`) from Anywhere.
   - Allow custom TCP port `8080` (or `5000` depending on your app configuration) if you want direct access to the container.
7. Click **Launch Instance**.

---

### 4. EC2 Instance Host Configuration

Once the instance is running, connect via SSH and install the required dependencies:

```bash
# Connect to your instance
ssh -i /path/to/your-key.pem ubuntu@<your-ec2-public-ip>

# Update and upgrade the system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
sudo apt-get install -y docker.io

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add the 'ubuntu' user to the docker group so you don't need 'sudo' for docker commands
sudo usermod -aG docker ubuntu
```

_Note: Run `newgrp docker` or log out and log back in to apply the group membership changes._

---

### 5. GitHub Repository Secrets Configuration

To link your repository to AWS and automate the CI/CD pipeline, add the following secrets in GitHub:

1. Go to your GitHub repository -> **Settings** -> **Secrets and variables** -> **Actions**.
2. Click **New repository secret** and add the following keys:
   - `AWS_ACCESS_KEY_ID`: Your IAM user's Access Key ID.
   - `AWS_SECRET_ACCESS_KEY`: Your IAM user's Secret Access Key.
   - `AWS_REGION`: The AWS region where ECR/EC2 are located (e.g., `us-east-1`).
   - `AWS_ECR_REPOSITORY_NAME`: The name of your ECR repository (e.g., `math-marks-prediction`).
   - `AWS_ECR_LOGIN_URI`: Your ECR host domain (e.g., `<aws_account_id>.dkr.ecr.<region>.amazonaws.com`).

---

### 6. GitHub Actions Self-Hosted Runner on EC2

To run the deployment commands directly on your EC2 instance (e.g. running the Docker container on port `8080`), you must configure the EC2 instance as a self-hosted runner:

1. Go to your GitHub repository -> **Settings** -> **Actions** -> **Runners**.
2. Click **New self-hosted runner** and select **Linux**.
3. Copy the download and configuration commands under **Download** and **Configure**, and run them on your EC2 server:

```bash
# Create a folder for the runner
mkdir actions-runner && cd actions-runner

# Download the latest runner package (replace with URL provided by GitHub)
curl -o actions-runner-linux-x64-3.x.x.tar.gz -L https://github.com/actions/runner/releases/download/...

# Extract the installer
tar xzf ./actions-runner-linux-x64-3.x.x.tar.gz

# Configure the runner (Enter repo URL and Token provided by GitHub UI when prompted)
./config.sh --url https://github.com/your-username/your-repo --token YOUR_TOKEN
```

4. During configuration:
   - Select the default runner group.
   - Set the runner name (e.g., `ec2-runner`).
   - Add labels (e.g., `self-hosted`, `ubuntu-latest`).
     - _Important:_ If you configure the runner to handle the `ubuntu-latest` label, ensure your workflow target matches this label, or update `.github/workflows/main.yaml` to specify `runs-on: self-hosted`.
5. Install and run the GitHub runner as a background system service:

```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

---

## 🚀 CI/CD Pipeline & Deployment Execution

Once all prerequisites are met, the workflow operates automatically as defined in `.github/workflows/main.yaml`:

### Pipeline Workflow Steps

1. **Continuous Integration:** Lints the codebase and runs unit tests.
2. **Continuous Delivery (Build & Push ECR Image):**
   - Configures AWS credentials from GitHub secrets.
   - Logs into Amazon ECR.
   - Builds the Docker image based on the [Dockerfile](file:///D:/Workspace/datascience-projects/math_marks_prediction/Dockerfile).
   - Tags the image as `latest` and pushes it to ECR.
3. **Continuous Deployment:**
   - Runs on the self-hosted EC2 runner.
   - Pulls the latest image from ECR.
   - Stops and removes any previously running container.
   - Executes `docker run -d -p 8080:8080` (or your configured port mapping) to run the container.
   - Prunes unused Docker objects using `docker system prune -f` to prevent disk space issues on EC2.

---

## 🌐 Production Server Optimization (Optional Nginx Reverse Proxy)

To access the application on port `80` (HTTP) with production-grade reverse proxying:

1. Install Nginx on your EC2 instance:
   ```bash
   sudo apt-get install nginx -y
   ```
2. Configure Nginx to forward port 80 to your Flask application container (e.g., running on port `8080`):
   Edit `/etc/nginx/sites-available/default`:

   ```nginx
   server {
       listen 80;
       server_name your-domain-or-ec2-public-ip;

       location / {
           proxy_pass http://127.0.0.1:8080;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       }
   }
   ```

3. Restart Nginx:
   ```bash
   sudo systemctl restart nginx
   ```
