terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# Your public key
resource "aws_key_pair" "default" {
  key_name   = "ece-600-key-terraform"
  public_key = file("${path.module}/ece-600-key.pub")
}

# Create a VPC
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  enable_dns_hostnames = true

  tags = {
    Name = "custom-vpc"
  }
}

# Create a public subnet
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "us-east-1a"
  map_public_ip_on_launch = true

  tags = {
    Name = "public-subnet"
  }
}

# Create an internet gateway
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "igw"
  }
}

# Create a route table
resource "aws_route_table" "rt" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = {
    Name = "public-route-table"
  }
}

# Associate route table with subnet
resource "aws_route_table_association" "rta" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.rt.id
}

# Security group allowing SSH from your IP
resource "aws_security_group" "ssh" {
  name        = "allow_ssh"
  description = "Allow SSH from specific IP"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["68.9.86.242/32"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Get latest Ubuntu 20.04 AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# EC2 instance with NVIDIA driver, CUDA, and PyTorch
resource "aws_instance" "g4dn_instance" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = "g4dn.2xlarge"
  key_name                    = aws_key_pair.default.key_name
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.ssh.id]
  associate_public_ip_address = true

  user_data = <<-EOF
              #!/bin/bash
              exec > /home/ubuntu/setup.log 2>&1
              set -x
              apt update -y && apt upgrade -y
              apt install -y build-essential dkms curl wget python3-pip

              wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
              mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
              curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add -
              add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

              apt update -y
              apt install -y cuda-drivers
              sleep 10

              pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
              update-alternatives --install /usr/bin/python python /usr/bin/python3 1

              echo "Setup complete on $(date)" >> /home/ubuntu/setup.log
            EOF

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  tags = {
    Name = "g4dn-pytorch-instance"
  }
}

output "instance_public_ip" {
  value       = aws_instance.g4dn_instance.public_ip
  description = "Public IP of the instance"
}
